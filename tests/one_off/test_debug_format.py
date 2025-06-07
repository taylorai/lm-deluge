#!/usr/bin/env python3

import json
from lm_deluge.prompt import Conversation, Message, Text, ToolResult, ToolCall

# Test the format conversion
conversation = Conversation(
    [
        Message("user", [Text("Click on something")]),
        Message(
            "assistant",
            [
                ToolCall(
                    id="call_456",
                    name="computer_click",
                    arguments={"x": 200, "y": 100, "button": "left"},
                )
            ],
        ),
        Message(
            "user",
            [
                ToolResult(
                    tool_call_id="call_456",
                    result={
                        "_computer_use_output": True,
                        "output": {
                            "type": "computer_screenshot",
                            "image_url": "data:image/png;base64,ABC123",
                        },
                        "acknowledged_safety_checks": [
                            {
                                "id": "sc_789",
                                "code": "malicious_instructions",
                                "message": "Test safety check",
                            }
                        ],
                    },
                )
            ],
        ),
    ]
)

# Convert and print
result = conversation.to_openai_responses()
print(json.dumps(result, indent=2))
