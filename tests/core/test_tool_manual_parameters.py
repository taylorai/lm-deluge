from lm_deluge.tool import Tool


def test_manual_parameters_optional_union():
    tool = Tool(
        name="maybe",
        description="Handles optional values",
        parameters={"maybe": str | None, "value": int | float},
        run=lambda maybe, value: (maybe, value),
    )

    maybe_schema = tool.parameters["maybe"]
    value_schema = tool.parameters["value"]

    assert "anyOf" in maybe_schema
    assert {"type": "string"} in maybe_schema["anyOf"]
    assert {"type": "null"} in maybe_schema["anyOf"]

    assert "anyOf" in value_schema
    assert {"type": "integer"} in value_schema["anyOf"]
    assert {"type": "number"} in value_schema["anyOf"]

    assert set(tool.required) == {"maybe", "value"}


def test_manual_parameters_optional_flag():
    tool = Tool(
        name="maybe_optional",
        description="Uses optional flag",
        parameters={"name": (str | None, {"optional": True})},
        run=lambda name=None: name,
    )

    assert tool.required == []
    schema = tool.parameters["name"]
    assert {"type": "null"} in schema.get("anyOf", [])
