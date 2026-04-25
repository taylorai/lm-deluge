"""Live smoke test for DeepSeek V4 (Pro and Flash) via native API and OpenRouter.

Skips entries whose API key env var is not set.
"""

import asyncio
import os

import dotenv

from lm_deluge import LLMClient
from lm_deluge.models import registry

dotenv.load_dotenv()


ENTRIES = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-pro-anthropic-compat",
    "deepseek-v4-flash-anthropic-compat",
    "deepseek-v4-pro-openrouter",
    "deepseek-v4-flash-openrouter",
]


async def _smoke(name: str) -> None:
    model = registry[name]
    if not os.getenv(model.api_key_env_var):
        print(f"  SKIP {name}: {model.api_key_env_var} not set")
        return
    llm = LLMClient(name, max_new_tokens=64)
    responses = await llm.process_prompts_async(
        ["Reply with exactly: SMOKE_OK"],
        return_completions_only=False,
        show_progress=False,
    )
    resp = responses[0]
    assert resp is not None and not resp.is_error, (
        f"{name} failed: {resp.error_message if resp else 'None'}"
    )
    assert resp.completion.strip(), f"{name} empty completion"
    assert resp.usage is not None, f"{name} no usage"
    assert resp.cost is not None and resp.cost > 0, (
        f"{name} expected cost>0, got {resp.cost}"
    )
    print(
        f"  PASS {name}: '{resp.completion.strip()[:40]}' "
        f"(in={resp.usage.input_tokens}, out={resp.usage.output_tokens}, "
        f"cost=${resp.cost:.6f})"
    )


async def main() -> None:
    failed = 0
    for name in ENTRIES:
        print(f"\n--- {name} ---")
        try:
            await _smoke(name)
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{'=' * 50}")
    if failed:
        print(f"{failed} failed out of {len(ENTRIES)}")
        raise SystemExit(1)
    print(f"All {len(ENTRIES)} checks OK (skips count as OK).")


if __name__ == "__main__":
    asyncio.run(main())
