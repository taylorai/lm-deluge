"""Deterministic tests for lm_deluge.decide — uses a local fake decision server."""

import asyncio
import os
import sys

from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.decide import (
    PROVIDERS,
    REGISTRY,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    _build_request,
    decide,
    decide_parallel_async,
    register_decision_model,
    register_decision_provider,
    serialize_questions,
)

QUESTIONS = {
    "is_urgent": Noul("Does this convey urgency?"),
    "department": Choice(
        "Which team should handle this?",
        {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
    ),
    "frustration": Score("How frustrated?", ["Calm", "Frustrated", "Very angry"]),
}


def fake_answers(questions: dict) -> dict:
    answers = {}
    for key, q in questions.items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.96}
        elif q["type"] == "choice":
            first = next(iter(q["criteria"]))
            answers[key] = {
                "type": "choice",
                "choice": first,
                "confidence": 0.9,
                "probabilities": {
                    k: (1.0 if k == first else 0.0) for k in q["criteria"]
                },
            }
        else:
            n = len(q["criteria"])
            answers[key] = {
                "type": "score",
                "score": 1.3,
                "confidence": 0.55,
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": {
                    str(i): (0.7 if i == 1 else 0.3 / (n - 1)) for i in range(n)
                },
            }
    return answers


class FakeServer:
    """Fake decision endpoint. `script` is a list of statuses to return in order
    (then 200 forever)."""

    def __init__(self, script: list[int] | None = None):
        self.script = list(script or [])
        self.requests: list[dict] = []
        self.auth_headers: list[str] = []

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        self.auth_headers.append(request.headers.get("Authorization", ""))
        if self.script:
            status = self.script.pop(0)
            if status == -1:
                return web.Response(status=200, text="not json")
            if status == -2:
                return web.json_response({"answers": []})
            if status != 200:
                return web.Response(
                    status=status,
                    text=f"error {status}",
                    headers={"retry-after": "0"} if status == 429 else {},
                )
        return web.json_response(
            {
                "model": body["model"] + "-20260917",
                "answers": fake_answers(body["questions"]),
                "usage": {"input_tokens": 1_000_000, "output_tokens": 40},
            }
        )

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/decide", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore
        return f"http://127.0.0.1:{port}/v1"

    async def stop(self):
        await self.runner.cleanup()


async def with_fake(script, fn):
    server = FakeServer(script)
    base = await server.start()
    register_decision_provider("fake", base, "FAKE_DECISION_KEY", path="/decide")
    register_decision_model("jev-fake", "fake", "jev-latest", input_cost=0.042)
    os.environ["FAKE_DECISION_KEY"] = "fake-key"
    try:
        return await fn(server)
    finally:
        await server.stop()
        REGISTRY.pop("jev-fake", None)
        PROVIDERS.pop("fake", None)


def test_registry():
    for model, info in REGISTRY.items():
        assert info["provider"] in PROVIDERS, model
        assert "name" in info and "input_cost" in info
    for name, p in PROVIDERS.items():
        assert p["api_base"].startswith("https://"), name
        assert p["path"].startswith("/") and p["api_key_env_var"], name
    assert REGISTRY["jev"]["provider"] == "typesafe"
    print("PASSED: registry")


def test_build_request():
    os.environ["TYPESAFE_API_KEY"] = "ts-key"
    qs = serialize_questions(QUESTIONS)
    url, headers, payload = _build_request("jev", {"ticket": "hi"}, qs, {})
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer ts-key"
    assert payload["model"] == "jev-latest"
    assert payload["state"] == {"ticket": "hi"}
    assert payload["questions"]["is_urgent"] == {
        "type": "noul",
        "instructions": "Does this convey urgency?",
    }

    os.environ["OPENROUTER_API_KEY"] = "or-key"
    url, _, payload = _build_request("jev-1.13-openrouter", "x", qs, {})
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert payload["model"] == "typesafe/jev-1.13"
    print("PASSED: request building")


def test_question_validation():
    assert Noul("q?", true="yes", false="no").to_dict()["criteria"] == {
        "true": "yes",
        "false": "no",
    }
    # raw dicts pass through
    raw = {"type": "noul", "instructions": "q?"}
    assert serialize_questions({"a": raw})["a"] is raw

    bad = [
        {},
        {"a": {"type": "maybe", "instructions": "q"}},
        {"a": {"type": "noul"}},
        {"a": Choice("q", {})},
        {"a": Choice("q", {str(i): "x" for i in range(256)})},
        {"a": Score("q", ["only one"])},
        {"a": Score("q", [str(i) for i in range(11)])},
    ]
    for qs in bad:
        try:
            serialize_questions(qs)  # type: ignore
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {qs}")
    print("PASSED: question validation")


def test_missing_key_fails_fast():
    os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        asyncio.run(decide("x", QUESTIONS, model="jev"))
    except ValueError as e:
        assert "TYPESAFE_API_KEY" in str(e)
    else:
        raise AssertionError("expected ValueError")
    print("PASSED: missing API key fails fast")


def test_end_to_end_parsing():
    async def run(server: FakeServer):
        resp = await decide("Payouts failing!", QUESTIONS, model="jev-fake")
        assert not resp.is_error, resp.error_message
        assert resp.model == "jev-latest-20260917"
        assert server.auth_headers == ["Bearer fake-key"]
        urgent = resp.answers["is_urgent"]
        assert isinstance(urgent, NoulAnswer) and urgent.value is True
        dept = resp.answers["department"]
        assert isinstance(dept, ChoiceAnswer) and dept.choice == "billing"
        frus = resp.answers["frustration"]
        assert isinstance(frus, ScoreAnswer)
        assert frus.score == 1.3 and frus.level == "Frustrated"
        assert resp.input_tokens == 1_000_000 and resp.output_tokens == 40
        assert resp.cost is not None and abs(resp.cost - 0.042) < 1e-9

    asyncio.run(with_fake([], run))
    print("PASSED: end-to-end parsing")


def test_parallel_order_and_per_state_questions():
    async def run(server: FakeServer):
        states = [f"ticket {i}" for i in range(20)]
        per_state = [
            {"pick": Choice("which?", {f"opt{i}": "x", "other": "y"})}
            for i in range(20)
        ]
        results = await decide_parallel_async(
            states, per_state, model="jev-fake", show_progress=False
        )
        assert [r.id for r in results] == list(range(20))
        for i, r in enumerate(results):
            assert r.state == f"ticket {i}"
            assert r.answers["pick"].choice == f"opt{i}"  # type: ignore

        try:
            await decide_parallel_async(states, per_state[:3], model="jev-fake")
        except ValueError:
            pass
        else:
            raise AssertionError("expected length mismatch error")

    asyncio.run(with_fake([], run))
    print("PASSED: parallel ordering + per-state questions")


def test_retries():
    async def run(server: FakeServer):
        resp = await decide("x", QUESTIONS, model="jev-fake", max_attempts=4)
        assert not resp.is_error, resp.error_message
        assert len(server.requests) == 3

    # 429 then 529 then success
    asyncio.run(with_fake([429, 529], run))
    print("PASSED: retries on 429/529")


def test_non_retryable_and_exhaustion():
    async def run_422(server: FakeServer):
        resp = await decide("x", QUESTIONS, model="jev-fake", max_attempts=5)
        assert resp.is_error and resp.status_code == 422
        assert len(server.requests) == 1

    asyncio.run(with_fake([422], run_422))

    async def run_all_429(server: FakeServer):
        resp = await decide("x", QUESTIONS, model="jev-fake", max_attempts=2)
        assert resp.is_error and resp.status_code == 429
        assert len(server.requests) == 2

    asyncio.run(with_fake([429, 429], run_all_429))

    async def run_malformed(server: FakeServer):
        resp = await decide("x", QUESTIONS, model="jev-fake")
        assert resp.is_error and "Malformed" in (resp.error_message or "")

    asyncio.run(with_fake([-1], run_malformed))
    asyncio.run(with_fake([-2], run_malformed))
    print("PASSED: non-retryable, exhaustion, malformed")


if __name__ == "__main__":
    test_registry()
    test_build_request()
    test_question_validation()
    test_missing_key_fails_fast()
    test_end_to_end_parsing()
    test_parallel_order_and_per_state_questions()
    test_retries()
    test_non_retryable_and_exhaustion()
    print("\nAll decide tests passed.")
