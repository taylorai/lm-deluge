"""The Responses `reasoning` object echoes request settings, not reasoning text.

A Luna tool round streamed through Bedrock came back with no reasoning output
item and `"reasoning": {"effort": "high", "summary": "detailed"}`; the parser
turned "detailed" into a Thinking part, which apps then showed as a plan step.
"""

from lm_deluge.api_requests.openai import parse_openai_responses_data
from lm_deluge.prompt import Thinking, ToolCall


def response(output):
    return {
        "status": "completed",
        "reasoning": {"effort": "high", "summary": "detailed"},
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "search",
    "arguments": '{"queries": ["horses"]}',
}


def test_settings_echo_is_not_thinking():
    content, thinking, _ = parse_openai_responses_data(response([CALL]))
    assert not any(isinstance(part, Thinking) for part in content.parts)
    assert thinking is None
    assert [part.name for part in content.parts if isinstance(part, ToolCall)] == [
        "search"
    ]


def test_reasoning_item_summary_is_thinking():
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Search sample clauses first."}],
        "encrypted_content": "opaque",
    }
    content, thinking, _ = parse_openai_responses_data(response([reasoning, CALL]))
    thoughts = [part for part in content.parts if isinstance(part, Thinking)]
    assert [part.content for part in thoughts] == ["Search sample clauses first."]
    assert thoughts[0].raw_payload == reasoning
    assert thinking == "Search sample clauses first."


if __name__ == "__main__":
    test_settings_echo_is_not_thinking()
    test_reasoning_item_summary_is_thinking()
    print("Responses reasoning settings tests passed")
