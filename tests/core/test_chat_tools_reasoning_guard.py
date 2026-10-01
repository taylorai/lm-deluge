"""
OpenAI reasoning models reject function tools on Chat Completions (with reasoning
enabled). The request config check must fail fast for both the direct OpenAI entries
and the Azure AI Foundry entries of the same models, and for aliases like `gpt-6`.
"""

from lm_deluge import Conversation, Tool
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams


def _noop(x: str) -> str:
    return x


TOOL = Tool.from_function(_noop)


def _context(
    model: str,
    *,
    effort: str | None = "low",
    use_responses_api: bool = False,
    tools: list | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name=model,
        prompt=Conversation().user("hi"),
        sampling_params=SamplingParams(reasoning_effort=effort),  # type: ignore[arg-type]
        tools=[TOOL] if tools is None else tools,
        use_responses_api=use_responses_api,
    )


def _rejects(ctx: RequestContext) -> bool:
    try:
        ctx.validate_request_config()
    except ValueError as e:
        assert "does not support function tools" in str(e), e
        return True
    return False


def test_rejects_tools_with_reasoning_on_chat_completions() -> None:
    for model in [
        "gpt-6-sol",
        "gpt-6",  # alias of gpt-6-sol
        "gpt-6-sol-azure",
        "gpt-6-luna-azure",
        "gpt-6.1-sol",
        "gpt-6.1-sol-azure",
        "gpt-5.6-sol",
        "gpt-5.6-luna-azure",
        "gpt-5.6-terra-azure",
    ]:
        assert _rejects(_context(model)), f"{model} should reject tools + reasoning"


def test_allows_supported_combinations() -> None:
    # Responses API supports tools with reasoning
    assert not _rejects(_context("gpt-6-sol-azure", use_responses_api=True))
    assert not _rejects(_context("gpt-6.1-sol-azure", use_responses_api=True))
    # GPT-6 Sol/Luna allow tools on Chat Completions when reasoning is disabled
    assert not _rejects(_context("gpt-6-sol-azure", effort="none"))
    assert not _rejects(_context("gpt-6-luna", effort="none"))
    # No tools: nothing to reject
    assert not _rejects(_context("gpt-6.1-sol-azure", tools=[]))
    # Non-reasoning Azure chat model is unaffected
    assert not _rejects(_context("gpt-chat-latest-azure"))


if __name__ == "__main__":
    test_rejects_tools_with_reasoning_on_chat_completions()
    test_allows_supported_combinations()
    print("All chat tools + reasoning guard tests passed.")
