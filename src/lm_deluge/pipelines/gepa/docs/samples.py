"""
GEPA-style population optimizer for fts-bench using lm-deluge (no litellm dependency).

Features:
- Maintains a pool of candidates with per-example validation scores.
- Selects a parent (best-by-val), mutates a single component, and accepts only if
  minibatch reward improves; accepted candidates get a full val eval and join the pool.
- Components: system_prompt, search_docstring, fetch_docstring.
- Rollouts are run via verifiers + OpenAI SDK (pointing to lm-deluge proxy server); reflection uses LLMClient.

Prerequisites:
    Start the lm-deluge proxy server first:
        python -m lm_deluge.server --port 8000

Run:
    uv run python gepa_lm_deluge_full.py --corpus-file ... --queries-file ... --env-file ...
"""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import verifiers as vf  # type: ignore
from datasets import Dataset  # type: ignore
from dotenv import load_dotenv
from fts_bench import (  # type: ignore
    DEFAULT_FETCH_DOCSTRING,
    DEFAULT_SEARCH_DOCSTRING,
    DEFAULT_SYSTEM_PROMPT,
)
from verifiers.utils.tool_utils import convert_func_to_oai_tool  # type: ignore

from openai import AsyncOpenAI  # type: ignore

from lm_deluge.client import LLMClient  # type: ignore
from lm_deluge.util.json import try_load_json  # type: ignore

# ---------------------- Helpers ---------------------- #


def _clean_state(state: dict[str, Any]) -> dict[str, Any]:
    drop = {"prompt", "completion", "responses"}
    return {k: v for k, v in state.items() if k not in drop}


def _extract_assistant_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _count_tool_calls(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            total += len(msg["tool_calls"])
    return total


def _summarize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact view of assistant tool calls (name + truncated args)."""
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            fn = (tc.get("function") or {}).get("name", "")
            assert fn, "tool call missing name"
            args_raw = (tc.get("function") or {}).get("arguments", "")
            args_str = str(args_raw)
            calls.append({"name": fn, "args": args_str})
    return calls


def _parse_documents_from_completion(messages: list[dict[str, Any]]) -> list[str]:
    assistant_msg = _extract_assistant_message(messages)
    if "{" in assistant_msg:
        assistant_msg = "{" + assistant_msg.split("{", 1)[1]
    parsed = try_load_json(assistant_msg)
    if isinstance(parsed, dict):
        docs = parsed.get("documents", [])
        if isinstance(docs, list):
            return [str(doc) for doc in docs]
    return []


def _question_key_from_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return "question"
    keys = records[0].keys()
    if "question" in keys:
        return "question"
    if "query" in keys:
        return "query"
    return "question"


def _format_dataset(
    env: vf.Environment,
    records: list[dict[str, Any]],
    system_prompt: str,
    question_key: str,
) -> Dataset:
    ds = Dataset.from_list(records)
    if "prompt" in ds.column_names:
        ds = ds.remove_columns("prompt")
    return env.format_dataset(
        ds,
        system_prompt=system_prompt,
        few_shot=env.few_shot,
        question_key=question_key,
    )


def _prepare_env(env: vf.ToolEnv, candidate: dict[str, str]) -> None:
    # Update text components and rebuild tool schemas.
    if env.tools:
        if len(env.tools) >= 1:
            env.tools[0].__doc__ = candidate["search_docstring"]
        if len(env.tools) >= 2:
            env.tools[1].__doc__ = candidate["fetch_docstring"]
        env.oai_tools = [convert_func_to_oai_tool(tool) for tool in env.tools]
        env.tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in env.tools
        }
    env.system_prompt = candidate["system_prompt"]


def _run_generate_sync(
    env: vf.Environment,
    dataset: Dataset,
    client: Any,
    model: str,
    max_concurrency: int,
    rollouts_per_example: int,
):
    async def _run():
        outputs: vf.GenerateOutputs = await env.generate(
            inputs=dataset,
            client=client,  # type: ignore[arg-type]
            model=model,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=max_concurrency,
            use_tqdm=False,
        )
        return outputs

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(_run(), loop).result()
        return loop.run_until_complete(_run())


@dataclass
class EvalResult:
    scores: list[float]
    trajectories: list[dict[str, Any]]
    avg_score: float
    example_ids: list[Any]
    subscores: dict[Any, float]


def evaluate_candidate(
    env: vf.ToolEnv,
    candidate: dict[str, str],
    records: list[dict[str, Any]],
    client: Any,
    model: str,
    max_concurrency: int,
    capture_traces: bool,
    rollouts_per_example: int,
    return_subscores: bool = False,
) -> EvalResult:
    _prepare_env(env, candidate)
    question_key = _question_key_from_records(records)
    formatted = _format_dataset(env, records, candidate["system_prompt"], question_key)
    results = _run_generate_sync(
        env,
        formatted,
        client,
        model,
        max_concurrency=max_concurrency,
        rollouts_per_example=rollouts_per_example,
    )

    trajectories: list[dict[str, Any]] = []
    scores = [float(r) for r in results.reward]
    example_ids: list[Any] = []
    subscores: dict[Any, float] = {}
    for idx in range(len(formatted)):
        completion_messages = results.completion[idx]
        ex_id = results.example_id[idx]
        example_ids.append(ex_id)
        if return_subscores:
            subscores[ex_id] = scores[idx]
        traj = {
            "example_id": ex_id,
            "question": formatted[idx].get(question_key, ""),
            "answer": str(results.answer[idx]),
            "reward": scores[idx],
            "tool_calls": _count_tool_calls(completion_messages),  # type: ignore
            "tool_calls_detail": _summarize_tool_calls(completion_messages),  # type: ignore
            "assistant_message": _extract_assistant_message(completion_messages),  # type: ignore
            "predicted_documents": _parse_documents_from_completion(
                completion_messages  # type: ignore
            ),
            "prompt_messages": results.prompt[idx],
            "completion_messages": completion_messages,
            "state": _clean_state(results.state[idx]),
        }
        trajectories.append(traj)

    avg_score = sum(scores) / max(len(scores), 1)
    if not capture_traces:
        trajectories = []
    if not return_subscores:
        subscores = {}
    return EvalResult(
        scores=scores,
        trajectories=trajectories,
        avg_score=avg_score,
        example_ids=example_ids,
        subscores=subscores,
    )


def _build_reflection_prompt(
    component: str, current_text: str, trajectories: list[dict[str, Any]], k: int = 4
) -> str:
    worst = sorted(trajectories, key=lambda t: t.get("reward", 0.0))[:k]
    intro = {
        "system_prompt": "Refine the system prompt for the search agent.",
        "search_docstring": "Refine the SEARCH tool description so the model issues higher-recall queries.",
        "fetch_docstring": "Refine the FETCH tool description so the model inspects the right snippets and returns correct doc IDs.",
    }[component]
    lines = [
        intro,
        f"Return ONLY the improved text for the {component.replace('_', ' ')}.",
        "",
        "Current text:",
        current_text,
        "",
        "Trajectories:",
    ]
    for t in worst:
        tool_calls_detail = t.get("tool_calls_detail", [])
        tool_calls_str = (
            "; ".join(
                [f"{c.get('name', '')}({c.get('args', '')})" for c in tool_calls_detail]
            )
            if tool_calls_detail
            else "none"
        )
        lines.append(
            f"- Q: {t.get('question', '')}\n"
            f"  Truth doc: {t.get('answer', '')}\n"
            f"  Predicted: {t.get('predicted_documents', [])}\n"
            f"  Reward: {t.get('reward', 0.0)} | Tool calls: {t.get('tool_calls', 0)} ({tool_calls_str})\n"
            f"  Assistant msg: {t.get('assistant_message', '')}\n"
        )
    lines.append(
        "Improve the component to boost recall/precision and ensure the final JSON includes correct doc IDs."
    )
    return "\n".join(lines)


def propose_new_text(
    reflection_client: LLMClient,  # type: ignore
    component: str,
    current_text: str,
    trajectories: list[dict[str, Any]],
) -> str:
    prompt = _build_reflection_prompt(component, current_text, trajectories)
    resp = reflection_client.process_prompts_sync([prompt], show_progress=False)[0]
    text = resp.completion.strip()
    return text if text else current_text


# ---------------------- Frontier / merge helpers ---------------------- #


def compute_val_frontier(population: list["CandidateRecord"]) -> dict[Any, set[int]]:
    per_val_best: dict[Any, set[int]] = {}
    max_score: dict[Any, float] = {}
    for idx, cand in enumerate(population):
        for val_id, score in cand.val_subscores.items():
            best = max_score.get(val_id)
            if best is None or score > best:
                max_score[val_id] = score
                per_val_best[val_id] = {idx}
            elif score == best:
                per_val_best[val_id].add(idx)
    return per_val_best


def frontier_union(frontier: dict[Any, set[int]]) -> set[int]:
    all_ids: set[int] = set()
    for ids in frontier.values():
        all_ids.update(ids)
    return all_ids


def choose_merge_parents(
    frontier_union_set: set[int],
    population: list["CandidateRecord"],
    rng: random.Random,
) -> tuple[int, int] | None:
    if len(frontier_union_set) < 2:
        return None
    choices = list(frontier_union_set)
    p1 = rng.choice(choices)
    choices.remove(p1)
    p2 = rng.choice(choices)
    return p1, p2


def merge_candidates(
    parent_a: dict[str, str],
    parent_b: dict[str, str],
    components: list[str],
    rng: random.Random,
) -> dict[str, str]:
    child = {}
    for comp in components:
        child[comp] = rng.choice([parent_a[comp], parent_b[comp]])
    return child


# ---------------------- GEPA loop ---------------------- #


@dataclass
class CandidateRecord:
    candidate: dict[str, str]
    val_scores: list[float]
    val_avg: float
    parents: list[int]
    val_subscores: dict[Any, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEPA-style optimizer for fts-bench using lm-deluge."
    )
    parser.add_argument(
        "--corpus-file", default="/Users/benjamin/building_codes_corpus.jsonl"
    )
    parser.add_argument(
        "--queries-file",
        default="/Users/benjamin/building_codes_queries_with_labels.jsonl",
    )
    parser.add_argument("--env-file", default="/Users/benjamin/Desktop/llm_tokens.env")
    parser.add_argument(
        "--model",
        default="claude-5-mini",
        help="Model for rollouts via lm-deluge proxy server.",
    )
    parser.add_argument(
        "--proxy-url",
        default="http://localhost:8000/v1",
        help="URL of the lm-deluge proxy server.",
    )
    parser.add_argument(
        "--reflection-model",
        default="gpt-4.1-mini",
        help="Model for reflection via LLMClient.",
    )
    parser.add_argument("--train-examples", type=int, default=48)
    parser.add_argument("--val-examples", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--minibatch-size", type=int, default=6)
    parser.add_argument(
        "--eval-every", type=int, default=5, help="Val evaluation cadence for logging."
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=1000,
        help="Budget in rollout evaluations.",
    )
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument(
        "--use-merge", action="store_true", help="Enable merge proposals."
    )
    parser.add_argument(
        "--max-merge-invocations", type=int, default=5, help="Max merge attempts."
    )
    parser.add_argument(
        "--merge-period",
        type=int,
        default=3,
        help="Try merge every N iters when merges remain.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    rng = random.Random(args.seed)

    # Build base environment once (keeps the index hot).
    base_env = vf.load_environment(
        "fts-bench",
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        max_turns=12,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        search_docstring=DEFAULT_SEARCH_DOCSTRING,
        fetch_docstring=DEFAULT_FETCH_DOCSTRING,
    )
    if base_env.dataset is None:
        raise ValueError("fts-bench environment did not return a dataset.")

    # Strip prompts to get raw records for re-formatting per candidate.
    if "prompt" in base_env.dataset.column_names:
        raw_ds = base_env.dataset.remove_columns("prompt")
    else:
        raw_ds = base_env.dataset

    train_ds = raw_ds.select(range(min(len(raw_ds), args.train_examples)))
    remaining_start = len(train_ds)
    val_end = min(len(raw_ds), remaining_start + args.val_examples)
    val_ds = (
        raw_ds.select(range(remaining_start, val_end))
        if val_end > remaining_start
        else train_ds
    )

    train_records = [train_ds[i] for i in range(len(train_ds))]
    val_records = [val_ds[i] for i in range(len(val_ds))]
    question_key = _question_key_from_records(train_records or val_records)  # noqa

    # Create OpenAI client pointing to lm-deluge proxy server
    rollout_client = AsyncOpenAI(base_url=args.proxy_url, api_key="not-needed")
    reflection_client = LLMClient(args.reflection_model, progress="tqdm")

    seed_candidate = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "search_docstring": DEFAULT_SEARCH_DOCSTRING,
        "fetch_docstring": DEFAULT_FETCH_DOCSTRING,
    }

    # Evaluate seed on val set.
    seed_eval = evaluate_candidate(
        base_env,
        seed_candidate,
        val_records,
        rollout_client,
        args.model,
        args.max_concurrency,
        capture_traces=False,
        rollouts_per_example=args.rollouts_per_example,
        return_subscores=True,
    )
    population: list[CandidateRecord] = [
        CandidateRecord(
            candidate=seed_candidate,
            val_scores=seed_eval.scores,
            val_avg=seed_eval.avg_score,
            parents=[],
            val_subscores=seed_eval.subscores,
        )
    ]
    best_idx = 0
    metric_calls = len(val_records) * args.rollouts_per_example
    print(
        f"Seed val avg reward: {seed_eval.avg_score:.3f} over {len(val_records)} examples"
    )

    components = ["system_prompt", "search_docstring", "fetch_docstring"]
    merges_due = 0
    merges_tested = 0
    frontier = compute_val_frontier(population)

    def print_rollout_usage(rollout_client: AsyncOpenAI):
        # Usage tracking not available via proxy - would need server-side tracking
        print("Rollout client: using lm-deluge proxy server")

    for it in range(1, args.iterations + 1):
        print(f"=== Starting iteration {it} ===")
        print_rollout_usage(rollout_client)
        # print(rollout_client._clients)
        if metric_calls >= args.max_metric_calls:
            print(f"Stopping: reached metric budget {metric_calls}")
            break

        # Attempt merge first if scheduled
        if (
            args.use_merge
            and merges_due > 0
            and merges_tested < args.max_merge_invocations
            and frontier_union(frontier)
        ):
            parent_pair = choose_merge_parents(
                frontier_union(frontier), population, rng
            )
            if parent_pair is not None:
                p1_idx, p2_idx = parent_pair
                parent_a = population[p1_idx].candidate
                parent_b = population[p2_idx].candidate

                minibatch = rng.sample(
                    train_records, k=min(args.minibatch_size, len(train_records))
                )
                eval_p1 = evaluate_candidate(
                    base_env,
                    parent_a,
                    minibatch,
                    rollout_client,
                    args.model,
                    args.max_concurrency,
                    capture_traces=False,
                    rollouts_per_example=args.rollouts_per_example,
                )
                eval_p2 = evaluate_candidate(
                    base_env,
                    parent_b,
                    minibatch,
                    rollout_client,
                    args.model,
                    args.max_concurrency,
                    capture_traces=False,
                    rollouts_per_example=args.rollouts_per_example,
                )
                metric_calls += 2 * len(minibatch) * args.rollouts_per_example

                child_candidate = merge_candidates(parent_a, parent_b, components, rng)
                eval_child = evaluate_candidate(
                    base_env,
                    child_candidate,
                    minibatch,
                    rollout_client,
                    args.model,
                    args.max_concurrency,
                    capture_traces=False,
                    rollouts_per_example=args.rollouts_per_example,
                )
                metric_calls += len(minibatch) * args.rollouts_per_example

                parent_max = max(sum(eval_p1.scores), sum(eval_p2.scores))
                child_sum = sum(eval_child.scores)
                improved = child_sum > parent_max
                print(
                    f"[Iter {it}][MERGE] parents {p1_idx},{p2_idx} child_sum={child_sum:.2f} "
                    f"parent_max={parent_max:.2f} -> {'ACCEPT' if improved else 'REJECT'} "
                    f"| metric_calls={metric_calls}"
                )

                if improved:
                    val_eval = evaluate_candidate(
                        base_env,
                        child_candidate,
                        val_records,
                        rollout_client,
                        args.model,
                        args.max_concurrency,
                        capture_traces=False,
                        rollouts_per_example=args.rollouts_per_example,
                        return_subscores=True,
                    )
                    metric_calls += len(val_records) * args.rollouts_per_example
                    population.append(
                        CandidateRecord(
                            candidate=child_candidate,
                            val_scores=val_eval.scores,
                            val_avg=val_eval.avg_score,
                            parents=[p1_idx, p2_idx],
                            val_subscores=val_eval.subscores,
                        )
                    )
                    merges_due = max(0, merges_due - 1)
                    merges_tested += 1
                    frontier = compute_val_frontier(population)
                    if val_eval.avg_score >= population[best_idx].val_avg:
                        best_idx = len(population) - 1
                else:
                    # rejected merge; leave merges_due unchanged so it can be retried later
                    pass

        # Parent selection: best by val avg.
        parent_idx = max(range(len(population)), key=lambda i: population[i].val_avg)
        parent = population[parent_idx].candidate
        component = components[(it - 1) % len(components)]

        # Minibatch for reflection.
        minibatch = rng.sample(
            train_records, k=min(args.minibatch_size, len(train_records))
        )
        eval_curr = evaluate_candidate(
            base_env,
            parent,
            minibatch,
            rollout_client,
            args.model,
            args.max_concurrency,
            capture_traces=True,
            rollouts_per_example=args.rollouts_per_example,
        )
        metric_calls += len(minibatch) * args.rollouts_per_example

        new_text = propose_new_text(
            reflection_client, component, parent[component], eval_curr.trajectories
        )
        candidate_new = dict(parent)
        candidate_new[component] = new_text

        eval_new = evaluate_candidate(
            base_env,
            candidate_new,
            minibatch,
            rollout_client,
            args.model,
            args.max_concurrency,
            capture_traces=False,
            rollouts_per_example=args.rollouts_per_example,
        )
        metric_calls += len(minibatch) * args.rollouts_per_example

        old_sum = sum(eval_curr.scores)
        new_sum = sum(eval_new.scores)
        improved = new_sum > old_sum
        print(
            f"[Iter {it}] parent {parent_idx} comp={component} old_sum={old_sum:.2f} new_sum={new_sum:.2f} -> "
            f"{'ACCEPT' if improved else 'REJECT'} | metric_calls={metric_calls}"
        )

        if not improved:
            continue

        # Full val eval for accepted candidate.
        val_eval = evaluate_candidate(
            base_env,
            candidate_new,
            val_records,
            rollout_client,
            args.model,
            args.max_concurrency,
            capture_traces=False,
            rollouts_per_example=args.rollouts_per_example,
            return_subscores=True,
        )
        metric_calls += len(val_records) * args.rollouts_per_example
        population.append(
            CandidateRecord(
                candidate=candidate_new,
                val_scores=val_eval.scores,
                val_avg=val_eval.avg_score,
                parents=[parent_idx],
                val_subscores=val_eval.subscores,
            )
        )
        if val_eval.avg_score >= population[best_idx].val_avg:
            best_idx = len(population) - 1
        frontier = compute_val_frontier(population)
        if args.use_merge and merges_tested < args.max_merge_invocations:
            merges_due = min(merges_due + 1, args.max_merge_invocations - merges_tested)

        if it % args.eval_every == 0:
            print(
                f"    Val avg {val_eval.avg_score:.3f} (best {population[best_idx].val_avg:.3f}, pool {len(population)})"
            )

    best = population[best_idx]
    out_dir = Path("debug_runs/gepa_lm_deluge_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_system_prompt.txt").write_text(
        best.candidate["system_prompt"], encoding="utf-8"
    )
    (out_dir / "best_search_docstring.txt").write_text(
        best.candidate["search_docstring"], encoding="utf-8"
    )
    (out_dir / "best_fetch_docstring.txt").write_text(
        best.candidate["fetch_docstring"], encoding="utf-8"
    )
    print(
        f"Done. Best val {best.val_avg:.3f} (pool {len(population)}, metric calls {metric_calls}). "
        f"Artifacts in {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()

# uv run python gepa_lm_deluge_full.py \
#     --use-merge --max-merge-invocations 5 --merge-period 3 \
#     --corpus-file /Users/benjamin/ccr_corpus.jsonl \
#     --queries-file /Users/benjamin/ccr_queries_with_labels.jsonl \
#     --env-file .env --model gpt-5-mini --reflection-model gpt-5-mini
