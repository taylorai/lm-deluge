#!/usr/bin/env python3
"""Test Anthropic Skills support."""

from lm_deluge import Skill
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation


def test_skill_class():
    """Test Skill class creation and serialization."""

    # Test Anthropic-managed skill
    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")
    assert skill.type == "anthropic"
    assert skill.skill_id == "xlsx"
    assert skill.version == "latest"

    anthropic_format = skill.for_anthropic()
    assert anthropic_format == {
        "type": "anthropic",
        "skill_id": "xlsx",
        "version": "latest",
    }

    # Test custom skill
    custom_skill = Skill(
        type="custom",
        skill_id="skill_01AbCdEfGhIjKlMnOpQrStUv",
        version="1759178010641129",
    )
    assert custom_skill.type == "custom"
    assert custom_skill.skill_id == "skill_01AbCdEfGhIjKlMnOpQrStUv"
    assert custom_skill.version == "1759178010641129"

    custom_format = custom_skill.for_anthropic()
    assert custom_format["type"] == "custom"
    assert custom_format["skill_id"] == "skill_01AbCdEfGhIjKlMnOpQrStUv"
    assert custom_format["version"] == "1759178010641129"

    print("Skill class test passed!")


def test_skill_in_anthropic_request():
    """Test that skills are correctly added to Anthropic requests."""

    # Create a mock model
    model = APIModel.from_registry("claude-3.5-haiku")

    # Create a skill
    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    # Create context with skills
    context = RequestContext(
        task_id=1,
        model_name="claude-3.5-haiku",
        prompt=Conversation().user("Create an Excel spreadsheet"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        skills=[skill],
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Check beta headers
    assert "anthropic-beta" in headers
    assert "code-execution-2025-08-25" in headers["anthropic-beta"]
    assert "skills-2025-10-02" in headers["anthropic-beta"]

    # Check container with skills
    assert "container" in request_json
    assert "skills" in request_json["container"]
    assert len(request_json["container"]["skills"]) == 1
    assert request_json["container"]["skills"][0] == {
        "type": "anthropic",
        "skill_id": "xlsx",
        "version": "latest",
    }

    # Check code_execution tool was automatically added
    assert "tools" in request_json
    has_code_exec = any(
        t.get("type", "").startswith("code_execution") for t in request_json["tools"]
    )
    assert has_code_exec, "code_execution tool should be automatically added"

    print("Skill in Anthropic request test passed!")


def test_multiple_skills():
    """Test multiple skills in a single request."""

    model = APIModel.from_registry("claude-3.5-haiku")

    skills = [
        Skill(type="anthropic", skill_id="xlsx", version="latest"),
        Skill(type="anthropic", skill_id="pptx", version="latest"),
        Skill(
            type="custom", skill_id="skill_01AbCdEfGhIjKlMnOpQrStUv", version="latest"
        ),
    ]

    context = RequestContext(
        task_id=1,
        model_name="claude-3.5-haiku",
        prompt=Conversation().user("Create a spreadsheet and presentation"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        skills=skills,
    )

    request_json, headers = _build_anthropic_request(model, context)

    assert len(request_json["container"]["skills"]) == 3

    # Check all skills are present
    skill_ids = [s["skill_id"] for s in request_json["container"]["skills"]]
    assert "xlsx" in skill_ids
    assert "pptx" in skill_ids
    assert "skill_01AbCdEfGhIjKlMnOpQrStUv" in skill_ids

    print("Multiple skills test passed!")


def test_skills_with_existing_code_execution_tool():
    """Test that code_execution tool is not duplicated if already present."""

    model = APIModel.from_registry("claude-3.5-haiku")

    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    # Include code_execution tool explicitly
    tools = [{"type": "code_execution_20250825", "name": "code_execution"}]

    context = RequestContext(
        task_id=1,
        model_name="claude-3.5-haiku",
        prompt=Conversation().user("Create a spreadsheet"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        tools=tools,
        skills=[skill],
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Count code_execution tools
    code_exec_count = sum(
        1
        for t in request_json["tools"]
        if t.get("type", "").startswith("code_execution")
    )
    assert (
        code_exec_count == 1
    ), f"Expected 1 code_execution tool, got {code_exec_count}"

    print("Skills with existing code_execution test passed!")


def test_skills_not_supported_by_openai():
    """Test that skills raise NotImplementedError for OpenAI provider."""
    from lm_deluge.api_requests.openai import OpenAIRequest

    context = RequestContext(
        task_id=1,
        model_name="gpt-4.1-mini",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        skills=[Skill(type="anthropic", skill_id="xlsx", version="latest")],
    )

    try:
        OpenAIRequest(context)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "Skills are only supported by Anthropic" in str(e)
        print("OpenAI skills error test passed!")


def test_skills_not_supported_by_gemini():
    """Test that skills raise NotImplementedError for Gemini provider."""
    from lm_deluge.api_requests.gemini import GeminiRequest

    context = RequestContext(
        task_id=1,
        model_name="gemini-2.0-flash",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        skills=[Skill(type="anthropic", skill_id="xlsx", version="latest")],
    )

    try:
        GeminiRequest(context)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "Skills are only supported by Anthropic" in str(e)
        print("Gemini skills error test passed!")


def test_skills_dict_format():
    """Test that skills can be passed as raw dicts."""

    model = APIModel.from_registry("claude-3.5-haiku")

    # Pass skills as dicts (not Skill objects)
    skills = [
        {"type": "anthropic", "skill_id": "docx", "version": "latest"},
    ]

    context = RequestContext(
        task_id=1,
        model_name="claude-3.5-haiku",
        prompt=Conversation().user("Create a document"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        skills=skills,
    )

    request_json, headers = _build_anthropic_request(model, context)

    assert request_json["container"]["skills"][0]["skill_id"] == "docx"
    print("Skills dict format test passed!")


def test_container_id_in_request():
    """Test that container_id is included in the request when provided."""

    model = APIModel.from_registry("claude-3.5-haiku")

    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    # Create context with container_id
    context = RequestContext(
        task_id=1,
        model_name="claude-3.5-haiku",
        prompt=Conversation().user("Continue working on the spreadsheet"),
        sampling_params=SamplingParams(max_new_tokens=1024),
        skills=[skill],
        container_id="container_01234567890abcdef",
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Check that container has both id and skills
    assert "container" in request_json
    assert request_json["container"]["id"] == "container_01234567890abcdef"
    assert len(request_json["container"]["skills"]) == 1

    print("Container ID in request test passed!")


if __name__ == "__main__":
    test_skill_class()
    print()
    test_skill_in_anthropic_request()
    print()
    test_multiple_skills()
    print()
    test_skills_with_existing_code_execution_tool()
    print()
    test_skills_not_supported_by_openai()
    print()
    test_skills_not_supported_by_gemini()
    print()
    test_skills_dict_format()
    print()
    test_container_id_in_request()
    print("\n All skills tests passed!")
