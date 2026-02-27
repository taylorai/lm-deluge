"""Modal one-off test for PybubbleSandbox + LLM agent loop.

Run from repository root:

    modal run tests/one_off/test_pybubble_modal_llm.py

Optional model override:

    modal run tests/one_off/test_pybubble_modal_llm.py --model gpt-4.1-mini

This test does all work in a Modal Linux container:
- Installs system deps (`bubblewrap`, `slirp4netns`, `curl`)
- Installs Python deps from this repo's `pyproject.toml` (`sandbox` extra)
- Injects local `lm_deluge` source with `add_local_python_source`
- Verifies outbound curl works from `PybubbleSandbox`
- Verifies an LLM can use sandbox tools to run curl

Required Modal secrets:
- `OPENAI_API_KEY` (key: `OPENAI_API_KEY`)
- `ANTHROPIC_API_KEY` (key: `ANTHROPIC_API_KEY`)
"""

import json
import os

import dotenv
import modal
from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sandbox.pybubble_sandbox import PybubbleSandbox

dotenv.load_dotenv()

app = modal.App("lm-deluge-pybubble-llm-test")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("bubblewrap", "slirp4netns", "curl")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["sandbox"])
    .add_local_python_source("lm_deluge")
)

OPENAI_SECRET = modal.Secret.from_name("OPENAI_API_KEY")
ANTHROPIC_SECRET = modal.Secret.from_name("ANTHROPIC_API_KEY")


def _resolve_model_and_api_env(model: str | None) -> tuple[str, str]:
    """Pick model + API key env var from explicit arg."""
    if model:
        if model.startswith("gpt-"):
            return model, "OPENAI_API_KEY"
        if model.startswith("claude-"):
            return model, "ANTHROPIC_API_KEY"
        raise ValueError(
            "Unsupported model prefix for this test. "
            "Use a model starting with 'gpt-' or 'claude-'."
        )

    return "gpt-4.1-mini", "OPENAI_API_KEY"


@app.function(image=IMAGE, timeout=900, secrets=[OPENAI_SECRET, ANTHROPIC_SECRET])
async def run_pybubble_llm_check(model_name: str, api_key_env: str) -> str:
    """Run a remote Linux smoke test for PybubbleSandbox + LLM tool use."""
    if not os.getenv(api_key_env):
        raise RuntimeError(
            f"Expected {api_key_env} in environment from Modal Secret, but it was missing."
        )

    async with PybubbleSandbox(network_access=True, outbound_access=True) as sandbox:
        curl_output = await sandbox._exec("curl -s https://httpbin.org/ip")
        try:
            parsed = json.loads(curl_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Direct curl in PybubbleSandbox did not return JSON. "
                f"Output: {curl_output}"
            ) from e

        if "origin" not in parsed:
            raise RuntimeError(
                "Direct curl succeeded but response missing 'origin'. "
                f"Output: {curl_output}"
            )

        client = LLMClient(
            model_names=model_name, max_new_tokens=512, request_timeout=120
        )
        conv = Conversation().user(
            "Use the bash tool once to run: curl -s https://httpbin.org/ip\n"
            "Return only the exact command output."
        )
        _, resp = await client.run_agent_loop(
            conv, tools=sandbox.get_tools(), max_rounds=6
        )

    if resp.is_error:
        raise RuntimeError(f"LLM request failed: {resp.error_message}")

    if not resp.completion:
        raise RuntimeError("LLM returned an empty completion")

    if "origin" not in resp.completion.lower():
        raise RuntimeError(
            "LLM completion did not include expected httpbin payload. "
            f"Completion: {resp.completion}"
        )

    return (
        "✅ PybubbleSandbox + LLM test passed\n"
        f"Model: {model_name}\n"
        f"Direct curl output: {curl_output}\n"
        f"LLM completion: {resp.completion}"
    )


@app.local_entrypoint()
def main(model: str = "") -> None:
    """Local entrypoint for modal run."""
    model_name, api_key_env = _resolve_model_and_api_env(model or None)
    print(
        "Starting remote Modal test with "
        f"model={model_name}, key_env={api_key_env}, app={app.name}"
    )
    result = run_pybubble_llm_check.remote(
        model_name=model_name,
        api_key_env=api_key_env,
    )
    print(result)
