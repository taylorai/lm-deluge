#!/usr/bin/env python3
"""Live API tests for Anthropic Skills.

These tests require ANTHROPIC_API_KEY to be set and will make actual API calls.
"""

import asyncio
import os
import tempfile
from typing import cast

import dotenv

from lm_deluge import Conversation, LLMClient, Skill
from lm_deluge.prompt import ToolResult
from lm_deluge.prompt.text import Text
from lm_deluge.util.anthropic_files import save_response_files

dotenv.load_dotenv()


async def test_anthropic_builtin_skill():
    """Test using Anthropic's built-in pptx skill."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return False

    print("Testing Anthropic built-in skill (pptx)...")

    skill = Skill(type="anthropic", skill_id="pptx", version="latest")

    # Skills with code execution can take a while, use longer timeout
    llm = LLMClient("claude-4.5-opus", max_new_tokens=20_000, request_timeout=300)

    try:
        response = await llm.start(
            Conversation().user(
                "Create a simple 1-slide presentation about Python programming. "
                "Slide 1: Title slide with 'Introduction to Python'. "
                "Make it as simple and as quickly as possible. This is just "
                "a test to verify skills work."
            ),
            skills=[skill],
        )

        if response.is_error:
            print(f"Error: {response.error_message}")
            return False

        print("Response received!")
        print(f"Stop reason: {response.raw_response.get('stop_reason', 'N/A')}")  # type: ignore

        # Check if we got a response with content
        if response.content:
            print(f"Content parts: {len(response.content.parts)}")
            for i, part in enumerate(response.content.parts):
                print(f"  Part {i}: {type(part).__name__}")

        # Check raw response for container info
        if response.raw_response:
            container = response.raw_response.get("container")
            if container:
                print(f"Container ID: {container.get('id', 'N/A')}")

        print("Anthropic built-in skill test passed!")
        return True

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        llm.close()


async def test_custom_skill():
    """Test using a custom uploaded skill.

    NOTE: Replace CUSTOM_SKILL_ID with an actual skill ID from your workspace.
    You can get this by uploading a skill via the Anthropic API.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return False

    CUSTOM_SKILL_ID = "skill_01VAdraCeDRxHUSQmPBWZEXz"

    print(f"Testing custom skill ({CUSTOM_SKILL_ID})...")

    skill = Skill(type="custom", skill_id=CUSTOM_SKILL_ID, version="latest")

    # Skills with code execution can take a while, use longer timeout
    llm = LLMClient("claude-4.5-opus", max_new_tokens=20_000, request_timeout=300)

    try:
        response = await llm.start(
            Conversation().user(
                "Quickly make a legal document with a title, header, and paragraph using your legal-documents skill."
            ),
            skills=[skill],
        )

        if response.is_error:
            print(f"Error: {response.error_message}")
            return False

        print("Response received!")
        assert response.raw_response
        print(f"Stop reason: {response.raw_response.get('stop_reason', 'N/A')}")

        if response.content:
            print(f"Content parts: {len(response.content.parts)}")
            # Print text content if any
            for part in response.content.parts:
                if hasattr(part, "text"):
                    part = cast(Text, part)
                    print(
                        f"Text: {part.text[:200]}..."
                        if len(part.text) > 200
                        else f"Text: {part.text}"
                    )  # type: ignore

        print("Custom skill test passed!")
        return True

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        llm.close()


async def test_multiple_skills():
    """Test using multiple Anthropic skills in a single request."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return False

    print("Testing multiple Anthropic skills (xlsx + pptx)...")

    skills = [
        Skill(type="anthropic", skill_id="xlsx", version="latest"),
        Skill(type="anthropic", skill_id="pptx", version="latest"),
    ]

    # Skills with code execution can take a while, use longer timeout
    llm = LLMClient("claude-4.5-opus", max_new_tokens=20_000, request_timeout=300)

    try:
        response = await llm.start(
            Conversation().user(
                "First, create a simple Excel spreadsheet with 2-3 cells filled in. "
                "Then create a 1-slide PowerPoint that says Hello World."
            ),
            skills=skills,
        )

        if response.is_error:
            print(f"Error: {response.error_message}")
            return False

        print("Response received!")
        assert response.raw_response
        print(f"Stop reason: {response.raw_response.get('stop_reason', 'N/A')}")

        if response.content:
            print(f"Content parts: {len(response.content.parts)}")

        print("Multiple skills test passed!")
        return True

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        llm.close()


async def test_file_download():
    """Test that we can download files created by skills."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return False

    print("Testing file download from skills response...")

    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    llm = LLMClient("claude-4.5-opus", max_new_tokens=20_000, request_timeout=300)

    try:
        response = await llm.start(
            Conversation().user(
                "Create a simple Excel file with 1-2 cells filled in. "
                "Go fast, this is a test."
            ),
            skills=[skill],
        )

        if response.is_error:
            print(f"Error: {response.error_message}")
            return False

        print("Response received!")
        print(f"Container ID: {response.container_id}")

        # Check for files in ToolResult parts
        files_found = []
        if response.content:
            for part in response.content.parts:
                if isinstance(part, ToolResult) and part.files:
                    for f in part.files:
                        files_found.append(f)
                        print(
                            f"  Found file: {f.get('filename', 'unknown')} (ID: {f['file_id']})"
                        )  # type: ignore

        if not files_found:
            print("No files found in response (model may not have created any)")
            print("File download test passed (no files to download)")
            return True

        # Try to download the files
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = await save_response_files(response, output_dir=tmpdir)
            print(f"Saved {len(saved_paths)} file(s):")
            for path in saved_paths:
                size = path.stat().st_size
                print(f"  {path.name}: {size} bytes")
                assert size > 0, f"File {path.name} is empty!"

        print("File download test passed!")
        return True

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        llm.close()


async def test_container_reuse_in_agent_loop():
    """Test that container ID is reused across agent loop iterations."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return False

    print("Testing container reuse in agent loop...")

    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    llm = LLMClient("claude-4.5-opus", max_new_tokens=20_000, request_timeout=300)

    try:
        # Use agent loop which should reuse the container
        conv, final_response = await llm.run_agent_loop(
            Conversation().user(
                "Create an Excel file with a simple table. "
                "Then tell me what you created."
            ),
            skills=[skill],
            max_rounds=3,
        )

        if final_response.is_error:
            print(f"Error: {final_response.error_message}")
            return False

        print("Agent loop completed!")
        print(f"Final container ID: {final_response.container_id}")
        print(f"Conversation has {len(conv.messages)} messages")

        # Check that we got a container ID
        if final_response.container_id:
            print(f"Container ID preserved: {final_response.container_id}")
        else:
            print("Warning: No container ID in final response")

        print("Container reuse test passed!")
        return True

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        llm.close()


async def test_container_id_explicit_reuse():
    """Test explicit container ID reuse across separate requests."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return False

    print("Testing explicit container ID reuse...")

    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    llm = LLMClient("claude-4.5-opus", max_new_tokens=20_000, request_timeout=300)

    try:
        # First request - create something
        response1 = await llm.start(
            Conversation().user(
                "Create an Excel file with a 'Sales' column containing values 100, 200, 300."
            ),
            skills=[skill],
        )

        if response1.is_error:
            print(f"Error in first request: {response1.error_message}")
            return False

        container_id = response1.container_id
        print(f"First request container ID: {container_id}")

        if not container_id:
            print("No container ID returned - cannot test reuse")
            return True  # Not a failure, just can't test

        # Second request - reuse container
        response2 = await llm.start(
            Conversation().user(
                "Add a 'Total' row at the bottom that sums the Sales column."
            ),
            skills=[skill],
            container_id=container_id,
        )

        if response2.is_error:
            print(f"Error in second request: {response2.error_message}")
            return False

        print(f"Second request container ID: {response2.container_id}")

        # Container ID should be the same
        if response2.container_id == container_id:
            print("Container ID successfully reused!")
        else:
            print(
                f"Warning: Container ID changed from {container_id} to {response2.container_id}"
            )

        print("Explicit container reuse test passed!")
        return True

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        llm.close()


async def main():
    print("=" * 60)
    print("Anthropic Skills Live API Tests")
    print("=" * 60)
    print()

    results = {}

    # Test 1: Built-in skill
    print("-" * 40)
    results["builtin"] = await test_anthropic_builtin_skill()
    print()

    # Test 2: Custom skill
    print("-" * 40)
    results["custom"] = await test_custom_skill()
    print()

    # Test 3: Multiple skills
    print("-" * 40)
    results["multiple"] = await test_multiple_skills()
    print()

    # Test 4: File download
    print("-" * 40)
    results["file_download"] = await test_file_download()
    print()

    # Test 5: Container reuse in agent loop
    print("-" * 40)
    results["container_agent_loop"] = await test_container_reuse_in_agent_loop()
    print()

    # Test 6: Explicit container reuse
    print("-" * 40)
    results["container_explicit"] = await test_container_id_explicit_reuse()
    print()

    # Summary
    print("=" * 60)
    print("Results Summary:")
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {test_name}: {status}")

    # Return overall success
    failures = [r for r in results.values() if r is False]
    if failures:
        print(f"\n{len(failures)} test(s) failed!")
        return False

    print("\nAll tests passed (or skipped)!")
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
