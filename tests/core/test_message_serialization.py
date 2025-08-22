import json

from lm_deluge.prompt import Message, ToolCall, ToolResult


def test_message_to_log_handles_unserialisable_arguments():
    tc = ToolCall(id="1", name="fn", arguments={"callable": len})
    msg = Message("assistant", [tc])
    data = msg.to_log()
    json.dumps(data)  # should not raise
    assert data["content"][0]["arguments"]["callable"] == repr(len)


def test_message_to_log_handles_unserialisable_results():
    tr = ToolResult(tool_call_id="1", result={"fn": max})
    msg = Message("assistant", [tr])
    data = msg.to_log()
    json.dumps(data)  # should not raise
    assert data["content"][0]["result"]["fn"] == repr(max)

