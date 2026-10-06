"""Parallel client for "decision" APIs (TypeSafe Jev and compatible providers).

Decision models take a ``state`` (text or JSON) plus a map of typed
``questions`` and return typed ``answers`` (a choice, a score, or a yes/no
probability) instead of generated text. The wire format is the same across
providers; only the endpoint, auth env var, and model name differ, so adding a
provider is a registry entry rather than a new request class.

    from lm_deluge.decide import Choice, Noul, Score, decide

    resp = await decide(
        "Help! My payouts have been failing for 3 days.",
        {
            "is_urgent": Noul("Does this convey urgency?"),
            "department": Choice(
                "Which team should handle this?",
                {"billing": "Payments, refunds", "technical": "Bugs, outages"},
            ),
            "frustration": Score("How frustrated is the customer?", ["Calm", "Annoyed", "Furious"]),
        },
    )
    resp.answers["department"].choice  # "technical"
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
from tqdm.auto import tqdm

from .api_requests.base import parse_retry_after
from .embed import _CostTracker, _wait_for_capacity
from .tracker import StatusTracker

# Providers that speak the decision wire format. The request URL is
# ``api_base + path``; the key is read from ``api_key_env_var`` at request time.
PROVIDERS: dict[str, dict[str, str]] = {
    "typesafe": {
        "api_base": "https://api.typesafe.ai/v1",
        "path": "/systemone",
        "api_key_env_var": "TYPESAFE_API_KEY",
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/alpha",
        "path": "/decisions",
        "api_key_env_var": "OPENROUTER_API_KEY",
    },
}

# Short model id -> provider + upstream model name. Output tokens are free on
# every provider so far, so only input cost is tracked (USD per 1M tokens).
REGISTRY: dict[str, dict[str, Any]] = {
    "jev": {
        "provider": "typesafe",
        "name": "jev-latest",
        "input_cost": 0.042,
    },
    "jev-openrouter": {
        "provider": "openrouter",
        "name": "~typesafe/jev-latest",
        "input_cost": 0.042,
    },
    "jev-1.13-openrouter": {
        "provider": "openrouter",
        "name": "typesafe/jev-1.13",
        "input_cost": 0.042,
    },
}

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Client errors that will fail identically on retry.
NON_RETRYABLE_STATUSES = {400, 401, 403, 404, 422}


def register_decision_provider(
    name: str,
    api_base: str,
    api_key_env_var: str,
    path: str = "/decisions",
):
    """Add (or replace) a provider that implements the decision API."""
    PROVIDERS[name] = {
        "api_base": api_base.rstrip("/"),
        "path": path,
        "api_key_env_var": api_key_env_var,
    }


def register_decision_model(
    model_id: str,
    provider: str,
    name: str,
    input_cost: float | None = None,
):
    """Add (or replace) a decision model under a short id."""
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown decision provider '{provider}'. "
            f"Register it first with register_decision_provider()."
        )
    REGISTRY[model_id] = {"provider": provider, "name": name, "input_cost": input_cost}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

Instructions = str | dict | list


@dataclass
class Noul:
    """Yes/no question. Answer is a calibrated probability that it's true."""

    instructions: Instructions
    true: str | None = None
    false: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        criteria = {k: v for k, v in (("true", self.true), ("false", self.false)) if v}
        if criteria:
            out["criteria"] = criteria
        return out


@dataclass
class Choice:
    """Pick one option. ``criteria`` maps option keys to descriptions."""

    instructions: Instructions
    criteria: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass
class Score:
    """Rate against ordered levels (lowest first)."""

    instructions: Instructions
    criteria: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


Question = Noul | Choice | Score | dict[str, Any]


def _serialize_question(key: str, question: Question) -> dict[str, Any]:
    q: dict[str, Any] = (
        question.to_dict() if isinstance(question, (Noul, Choice, Score)) else question
    )
    qtype = q.get("type")
    if "instructions" not in q:
        raise ValueError(f"Question '{key}' is missing 'instructions'")
    if qtype == "noul":
        pass
    elif qtype == "choice":
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"Choice question '{key}' needs a non-empty criteria dict")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"Choice question '{key}' has {len(criteria)} options "
                f"(max {MAX_CHOICE_OPTIONS})"
            )
    elif qtype == "score":
        criteria = q.get("criteria")
        if not isinstance(criteria, list) or not (
            MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
        ):
            raise ValueError(
                f"Score question '{key}' needs a criteria list of "
                f"{MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels"
            )
    else:
        raise ValueError(
            f"Question '{key}' has unknown type {qtype!r} "
            "(expected 'noul', 'choice', or 'score')"
        )
    return q


def serialize_questions(questions: dict[str, Question]) -> dict[str, dict[str, Any]]:
    if not questions:
        raise ValueError("At least one question is required")
    return {k: _serialize_question(k, q) for k, q in questions.items()}


# ---------------------------------------------------------------------------
# Answers / responses
# ---------------------------------------------------------------------------


@dataclass
class NoulAnswer:
    noul: float
    type: Literal["noul"] = "noul"

    @property
    def value(self) -> bool:
        return self.noul >= 0.5


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None
    type: Literal["choice"] = "choice"


@dataclass
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    legend: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    type: Literal["score"] = "score"

    @property
    def level(self) -> Any:
        """Description of the most likely level."""
        if not self.probabilities:
            return None
        best = max(self.probabilities, key=lambda k: self.probabilities[k])
        return self.legend.get(best)


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def _parse_answer(raw: dict[str, Any]) -> Answer:
    atype = raw.get("type")
    if atype == "noul":
        return NoulAnswer(noul=raw["noul"])
    if atype == "choice":
        return ChoiceAnswer(
            choice=raw["choice"],
            probabilities=raw.get("probabilities") or {},
            confidence=raw.get("confidence"),
        )
    if atype == "score":
        return ScoreAnswer(
            score=raw["score"],
            probabilities=raw.get("probabilities") or {},
            legend=raw.get("legend") or {},
            confidence=raw.get("confidence"),
        )
    raise ValueError(f"Unknown answer type {atype!r}")


@dataclass
class DecisionResponse:
    id: int
    status_code: int | None
    is_error: bool
    error_message: str | None
    state: Any
    model: str | None = None
    answers: dict[str, Answer] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    raw: dict[str, Any] | None = None


def _get_model_info(model: str) -> tuple[dict[str, Any], dict[str, str]]:
    if model not in REGISTRY:
        raise ValueError(
            f"Unknown decision model '{model}'. Available: {', '.join(REGISTRY.keys())}"
        )
    info = REGISTRY[model]
    if info["provider"] not in PROVIDERS:
        raise ValueError(f"Unknown decision provider '{info['provider']}'")
    return info, PROVIDERS[info["provider"]]


def _build_request(
    model: str,
    state: Any,
    questions: dict[str, dict[str, Any]],
    extra_params: dict[str, Any],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build URL, headers, and payload for a decision request."""
    info, provider = _get_model_info(model)
    url = provider["api_base"] + provider["path"]
    api_key = os.environ.get(provider["api_key_env_var"])
    if not api_key:
        raise ValueError(
            f"{provider['api_key_env_var']} is not set (needed for model '{model}')"
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": info["name"],
        "state": state,
        "questions": questions,
        **extra_params,
    }
    return url, headers, payload


def _parse_response(
    task_id: int, state: Any, result: dict[str, Any], input_cost: float | None
) -> DecisionResponse:
    usage = result.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0) or 0
    return DecisionResponse(
        id=task_id,
        status_code=200,
        is_error=False,
        error_message=None,
        state=state,
        model=result.get("model"),
        answers={k: _parse_answer(v) for k, v in result["answers"].items()},
        input_tokens=input_tokens,
        output_tokens=usage.get("output_tokens", 0) or 0,
        cost=(
            input_tokens * input_cost / 1_000_000 if input_cost is not None else None
        ),
        raw=result,
    )


def _estimate_tokens(state: Any, questions: dict[str, Any]) -> int:
    text = state if isinstance(state, str) else json.dumps(state)
    return max((len(text) + len(json.dumps(questions))) // 4, 1)


async def _decide_one(
    task_id: int,
    state: Any,
    questions: dict[str, dict[str, Any]],
    model: str,
    extra_params: dict[str, Any],
    status_tracker: StatusTracker,
    capacity_lock: asyncio.Lock,
    max_requests_per_minute: int,
    max_attempts: int,
    request_timeout: int,
    cost_tracker: _CostTracker,
    pbar: tqdm | None,
) -> DecisionResponse:
    """Run a single decision request with retries, rate limiting, and concurrency control."""

    def error(status: int | None, message: str) -> DecisionResponse:
        status_tracker.num_tasks_failed += 1
        if pbar:
            pbar.update(1)
        return DecisionResponse(
            id=task_id,
            status_code=status,
            is_error=True,
            error_message=message,
            state=state,
        )

    url, headers, payload = _build_request(model, state, questions, extra_params)
    input_cost = REGISTRY[model]["input_cost"]
    # Cap so an oversized input can't wait forever on TPM capacity; the API
    # will reject it if it's genuinely too long.
    estimated_tokens = min(
        _estimate_tokens(state, questions), status_tracker.max_tokens_per_minute
    )

    status: int | None = None
    message = "Exhausted all attempts"
    for attempt in range(max_attempts):
        retry = attempt > 0
        await _wait_for_capacity(
            status_tracker,
            capacity_lock,
            estimated_tokens,
            max_requests_per_minute,
            retry=retry,
        )
        # check_capacity only bumps num_tasks_in_progress on the first attempt;
        # we release the slot on every failure, so re-take it on retries.
        if retry:
            status_tracker.num_tasks_in_progress += 1

        try:
            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    url, json=payload, headers=headers, allow_redirects=False
                ) as response,
            ):
                status = response.status
                if status == 200:
                    try:
                        result = await response.json(content_type=None)
                        parsed = _parse_response(task_id, state, result, input_cost)
                    except (AttributeError, KeyError, TypeError, ValueError) as e:
                        status_tracker.num_tasks_in_progress -= 1
                        return error(
                            200, f"Malformed response: {type(e).__name__}: {e}"
                        )
                    await cost_tracker.record(parsed.input_tokens)
                    status_tracker.task_succeeded(task_id)
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(cost_tracker.summary())
                    return parsed

                message = await response.text()
                status_tracker.num_tasks_in_progress -= 1
                if status in NON_RETRYABLE_STATUSES:
                    return error(status, message)
                if status == 429:
                    status_tracker.rate_limit_exceeded(parse_retry_after(response))
                    continue
                # 5xx / 529 overloaded: back off and retry
                if attempt < max_attempts - 1:
                    await asyncio.sleep(min(2**attempt, 16))
        except asyncio.TimeoutError:
            status_tracker.num_tasks_in_progress -= 1
            status, message = None, "Request timed out"
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
        except aiohttp.ClientError as e:
            status_tracker.num_tasks_in_progress -= 1
            status, message = None, f"{type(e).__name__}: {e}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))

    return error(status, message)


async def decide_parallel_async(
    states: list[Any],
    questions: dict[str, Question] | list[dict[str, Question]],
    model: str = "jev",
    max_attempts: int = 5,
    max_requests_per_minute: int = 3_000,
    max_tokens_per_minute: int = 10_000_000,
    max_concurrent_requests: int = 64,
    request_timeout: int = 30,
    show_progress: bool = True,
    **kwargs,
) -> list[DecisionResponse]:
    """Ask the same questions (or per-state questions) about many states in parallel.

    Args:
        states: Inputs to evaluate; each is a string, dict, or list.
        questions: Either one question map applied to every state, or a list
            of question maps (same length as ``states``).
        model: Decision model id (see REGISTRY).
        **kwargs: Extra top-level fields passed through to the API payload.

    Returns:
        One DecisionResponse per state, in input order.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    if max_requests_per_minute <= 0:
        raise ValueError("max_requests_per_minute must be > 0")
    if max_tokens_per_minute <= 0:
        raise ValueError("max_tokens_per_minute must be > 0")
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    if not states:
        return []

    _, provider = _get_model_info(model)
    if not os.environ.get(provider["api_key_env_var"]):
        raise ValueError(
            f"{provider['api_key_env_var']} is not set (needed for model '{model}')"
        )
    if isinstance(questions, list):
        if len(questions) != len(states):
            raise ValueError(
                f"Got {len(questions)} question maps for {len(states)} states"
            )
        per_state = [serialize_questions(q) for q in questions]
    else:
        shared = serialize_questions(questions)
        per_state = [shared] * len(states)

    cost_tracker = _CostTracker(cost_per_million=REGISTRY[model]["input_cost"] or 0.0)
    status_tracker = StatusTracker(
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        use_progress_bar=False,  # we manage our own tqdm
    )
    capacity_lock = asyncio.Lock()
    pbar = (
        tqdm(total=len(states), desc=f"Deciding [{model}]") if show_progress else None
    )

    results = await asyncio.gather(
        *[
            _decide_one(
                task_id=i,
                state=state,
                questions=qs,
                model=model,
                extra_params=kwargs,
                status_tracker=status_tracker,
                capacity_lock=capacity_lock,
                max_requests_per_minute=max_requests_per_minute,
                max_attempts=max_attempts,
                request_timeout=request_timeout,
                cost_tracker=cost_tracker,
                pbar=pbar,
            )
            for i, (state, qs) in enumerate(zip(states, per_state))
        ]
    )

    if pbar:
        pbar.close()

    if show_progress:
        parts = [f"Decided {len(states)} states"]
        if cost_tracker.total_tokens > 0:
            parts.append(f"{cost_tracker.total_tokens:,} input tokens")
        if cost_tracker.total_cost > 0:
            parts.append(f"${cost_tracker.total_cost:.6f}")
        if status_tracker.num_tasks_failed > 0:
            parts.append(f"{status_tracker.num_tasks_failed} failed")
        if status_tracker.num_rate_limit_errors > 0:
            parts.append(f"{status_tracker.num_rate_limit_errors} rate limited")
        print("  " + " | ".join(parts))

    return list(results)


async def decide(
    state: Any,
    questions: dict[str, Question],
    model: str = "jev",
    **kwargs,
) -> DecisionResponse:
    """Evaluate a single state. Same options as decide_parallel_async."""
    kwargs.setdefault("show_progress", False)
    results = await decide_parallel_async([state], questions, model=model, **kwargs)
    return results[0]


def decide_sync(
    states: list[Any],
    questions: dict[str, Question] | list[dict[str, Question]],
    model: str = "jev",
    **kwargs,
) -> list[DecisionResponse]:
    """Synchronous wrapper around decide_parallel_async."""
    return asyncio.run(decide_parallel_async(states, questions, model=model, **kwargs))
