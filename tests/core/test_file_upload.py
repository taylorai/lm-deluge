#!/usr/bin/env python3

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from lm_deluge import LLMClient
from lm_deluge.file import File
from lm_deluge.prompt import Conversation, Message

# Load environment variables from .env file
load_dotenv()


async def test_openai_file_upload():
    """Test file upload to OpenAI Files API"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for OpenAI upload.")
            temp_path = f.name

        # Create File object and upload
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")
        remote_file = await file.as_remote("openai")

        # Validate the result
        assert remote_file.is_remote
        assert remote_file.remote_provider == "openai"
        assert remote_file.file_id is not None
        assert remote_file.file_id.startswith("file-")
        print(f"✓ OpenAI file upload test passed (file_id: {remote_file.file_id})")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ OpenAI file upload test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_anthropic_file_upload():
    """Test file upload to Anthropic Files API"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for Anthropic upload.")
            temp_path = f.name

        # Create File object and upload
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")
        remote_file = await file.as_remote("anthropic")

        # Validate the result
        assert remote_file.is_remote
        assert remote_file.remote_provider == "anthropic"
        assert remote_file.file_id is not None
        print(f"✓ Anthropic file upload test passed (file_id: {remote_file.file_id})")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Anthropic file upload test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_google_file_upload():
    """Test file upload to Google Gemini Files API"""
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set, skipping test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for Google upload.")
            temp_path = f.name

        # Create File object and upload
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")
        remote_file = await file.as_remote("google")

        # Validate the result
        assert remote_file.is_remote
        assert remote_file.remote_provider == "google"
        assert remote_file.file_id is not None
        print(f"✓ Google file upload test passed (file_id: {remote_file.file_id})")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Google file upload test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_provider_validation():
    """Test that provider validation works in emission functions"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping provider validation test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file for provider validation.")
            temp_path = f.name

        # Upload to OpenAI
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")
        openai_file = await file.as_remote("openai")

        # Try to emit as Anthropic (should raise error)
        try:
            openai_file.anthropic()
            print("✗ Provider validation test failed: expected ValueError")
            os.remove(temp_path)
            return False
        except ValueError as e:
            assert "Cannot emit file uploaded to openai as Anthropic format" in str(e)
            print("✓ Provider validation test passed (OpenAI->Anthropic blocked)")

        # Verify OpenAI emission still works
        oa_result = openai_file.oa_chat()
        assert oa_result["type"] == "file"
        assert oa_result["file"]["file_id"] == openai_file.file_id
        print("✓ OpenAI emission works correctly")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Provider validation test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_already_remote_same_provider():
    """Test that uploading an already remote file with same provider returns itself"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping same provider test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file for same provider check.")
            temp_path = f.name

        # Upload to OpenAI
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")
        remote_file1 = await file.as_remote("openai")
        file_id1 = remote_file1.file_id

        # Try to upload again to same provider
        remote_file2 = await remote_file1.as_remote("openai")

        # Should be the same file
        assert remote_file2.file_id == file_id1
        print("✓ Same provider test passed (returns same file)")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Same provider test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_cross_provider_upload_blocked():
    """Test that trying to upload a remote file to different provider raises error"""
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        print("Both API keys not set, skipping cross-provider test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file for cross-provider check.")
            temp_path = f.name

        # Upload to OpenAI
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")
        openai_file = await file.as_remote("openai")

        # Try to upload to Anthropic (should raise error)
        try:
            await openai_file.as_remote("anthropic")
            print("✗ Cross-provider test failed: expected ValueError")
            os.remove(temp_path)
            return False
        except ValueError as e:
            assert "already uploaded to openai" in str(e)
            print("✓ Cross-provider upload blocked correctly")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Cross-provider test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_local_file_emission():
    """Test that local files can still be emitted to any provider"""
    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file for local emission.")
            temp_path = f.name

        # Create local file (not uploaded)
        file = File(data=Path(temp_path), media_type="text/plain", filename="test.txt")

        # Should be able to emit to both providers
        oa_result = file.oa_chat()
        assert oa_result["type"] == "file"
        assert "file_data" in oa_result["file"]

        anthropic_result = file.anthropic()
        assert anthropic_result["type"] == "document"
        assert anthropic_result["source"]["type"] == "base64"

        print("✓ Local file emission test passed")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Local file emission test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_openai_llm_call_with_uploaded_file():
    """Test making an actual LLM call with an uploaded file to OpenAI"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping OpenAI LLM test")
        return True

    try:
        # Use the sample.pdf file
        sample_pdf = Path(__file__).parent / "sample.pdf"
        if not sample_pdf.exists():
            print(f"✗ sample.pdf not found at {sample_pdf}")
            return False

        # Upload the file
        file = File(data=sample_pdf, media_type="application/pdf")
        remote_file = await file.as_remote("openai")
        print(f"Uploaded file to OpenAI: {remote_file.file_id}")

        # Create a message with the uploaded file
        msg = Message.user("What's in this file?")
        msg.parts.append(remote_file)
        conv = Conversation([msg])

        # Make the LLM call
        client = LLMClient("gpt-4o-mini")
        results = await client.process_prompts_async(prompts=[conv])

        if not results or len(results) == 0:
            print("✗ No results from LLM call")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ LLM call error: {result.error_message}")
            return False

        # Check that the response mentions "oakland" (case-insensitive)
        response_text = (
            result.content.completion.lower() if result.content.completion else ""
        )
        if "oakland" not in response_text:
            print("✗ Response does not mention 'oakland'")
            print(f"Full response: {response_text}")
            return False

        print("✓ OpenAI LLM call with uploaded file passed (mentions 'oakland')")
        return True

    except Exception as e:
        print(f"✗ OpenAI LLM test failed: {e}")
        return False


async def test_anthropic_llm_call_with_uploaded_file():
    """Test making an actual LLM call with an uploaded file to Anthropic"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping Anthropic LLM test")
        return True

    try:
        # Use the sample.pdf file
        sample_pdf = Path(__file__).parent / "sample.pdf"
        if not sample_pdf.exists():
            print(f"✗ sample.pdf not found at {sample_pdf}")
            return False

        # Upload the file
        file = File(data=sample_pdf, media_type="application/pdf")
        remote_file = await file.as_remote("anthropic")
        print(f"Uploaded file to Anthropic: {remote_file.file_id}")

        # Create a message with the uploaded file
        msg = Message.user("What's in this file?")
        msg.parts.append(remote_file)
        conv = Conversation([msg])

        # Make the LLM call
        client = LLMClient("claude-3.5-haiku")
        results = await client.process_prompts_async(prompts=[conv])

        if not results or len(results) == 0:
            print("✗ No results from LLM call")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ LLM call error: {result.error_message}")
            return False

        # Check that the response mentions "oakland" (case-insensitive)
        response_text = (
            result.content.completion.lower() if result.content.completion else ""
        )
        if "oakland" not in response_text:
            print("✗ Response does not mention 'oakland'")
            print(f"Full response: {response_text}")
            return False

        print("✓ Anthropic LLM call with uploaded file passed (mentions 'oakland')")
        return True

    except Exception as e:
        print(f"✗ Anthropic LLM test failed: {e}")
        return False


async def test_google_llm_call_with_uploaded_file():
    """Test making an actual LLM call with an uploaded file to Google Gemini"""
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set, skipping Google LLM test")
        return True

    try:
        # Use the sample.pdf file
        sample_pdf = Path(__file__).parent / "sample.pdf"
        if not sample_pdf.exists():
            print(f"✗ sample.pdf not found at {sample_pdf}")
            return False

        # Upload the file
        file = File(data=sample_pdf, media_type="application/pdf")
        remote_file = await file.as_remote("google")
        print(f"Uploaded file to Google: {remote_file.file_id}")

        # Create a message with the uploaded file
        msg = Message.user("What's in this file?")
        msg.parts.append(remote_file)
        conv = Conversation([msg])

        # Make the LLM call using a Gemini model
        client = LLMClient("gemini-2.0-flash")
        results = await client.process_prompts_async(prompts=[conv])

        if not results or len(results) == 0:
            print("✗ No results from LLM call")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ LLM call error: {result.error_message}")
            return False

        # Check that the response mentions "oakland" (case-insensitive)
        response_text = (
            result.content.completion.lower() if result.content.completion else ""
        )
        if "oakland" not in response_text:
            print("✗ Response does not mention 'oakland'")
            print(f"Full response: {response_text}")
            return False

        print("✓ Google LLM call with uploaded file passed (mentions 'oakland')")
        return True

    except Exception as e:
        print(f"✗ Google LLM test failed: {e}")
        return False


async def test_openai_responses_api_with_uploaded_file():
    """Test making an actual LLM call with uploaded file using OpenAI Responses API"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping OpenAI Responses API test")
        return True

    try:
        # Use the sample.pdf file
        sample_pdf = Path(__file__).parent / "sample.pdf"
        if not sample_pdf.exists():
            print(f"✗ sample.pdf not found at {sample_pdf}")
            return False

        # Upload the file
        file = File(data=sample_pdf, media_type="application/pdf")
        remote_file = await file.as_remote("openai")
        print(f"Uploaded file to OpenAI for Responses API: {remote_file.file_id}")

        # Create a message with the uploaded file
        msg = Message.user("What's in this file?")
        msg.parts.append(remote_file)
        conv = Conversation([msg])

        # Make the LLM call using Responses API with gpt-5-mini
        # Note: use_responses_api is set at the client level, not per-request
        client = LLMClient("gpt-5-mini", use_responses_api=True)
        results = await client.process_prompts_async(prompts=[conv])

        if not results or len(results) == 0:
            print("✗ No results from LLM call")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ LLM call error: {result.error_message}")
            return False

        # Check that the response mentions "oakland" (case-insensitive)
        response_text = (
            result.content.completion.lower() if result.content.completion else ""
        )
        if "oakland" not in response_text:
            print("✗ Response does not mention 'oakland'")
            print(f"Full response: {response_text}")
            return False

        print(
            "✓ OpenAI Responses API call with uploaded file passed (mentions 'oakland')"
        )
        return True

    except Exception as e:
        print(f"✗ OpenAI Responses API test failed: {e}")
        return False


async def test_delete_openai_file():
    """Test deleting an uploaded file from OpenAI"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping OpenAI delete test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for OpenAI deletion.")
            temp_path = f.name

        # Upload the file
        file = File(
            data=Path(temp_path), media_type="text/plain", filename="test_delete.txt"
        )
        remote_file = await file.as_remote("openai")
        file_id = remote_file.file_id

        # Delete the file
        deleted = await remote_file.delete()
        assert deleted

        print(f"✓ OpenAI file deletion test passed (deleted {file_id})")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ OpenAI file deletion test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_delete_anthropic_file():
    """Test deleting an uploaded file from Anthropic"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping Anthropic delete test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for Anthropic deletion.")
            temp_path = f.name

        # Upload the file
        file = File(
            data=Path(temp_path), media_type="text/plain", filename="test_delete.txt"
        )
        remote_file = await file.as_remote("anthropic")
        file_id = remote_file.file_id

        # Delete the file
        deleted = await remote_file.delete()
        assert deleted

        print(f"✓ Anthropic file deletion test passed (deleted {file_id})")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Anthropic file deletion test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def test_delete_google_file():
    """Test deleting an uploaded file from Google"""
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set, skipping Google delete test")
        return True

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file for Google deletion.")
            temp_path = f.name

        # Upload the file
        file = File(
            data=Path(temp_path), media_type="text/plain", filename="test_delete.txt"
        )
        remote_file = await file.as_remote("google")
        file_id = remote_file.file_id

        # Delete the file
        deleted = await remote_file.delete()
        assert deleted

        print(f"✓ Google file deletion test passed (deleted {file_id})")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        print(f"✗ Google file deletion test failed: {e}")
        if "temp_path" in locals():
            os.remove(temp_path)
        return False


async def main():
    print("Running file upload tests...\n")

    results = []
    results.append(await test_openai_file_upload())
    results.append(await test_anthropic_file_upload())
    results.append(await test_google_file_upload())
    results.append(await test_provider_validation())
    results.append(await test_already_remote_same_provider())
    results.append(await test_cross_provider_upload_blocked())
    results.append(await test_local_file_emission())

    print("\nRunning live LLM tests with uploaded files...\n")
    results.append(await test_openai_llm_call_with_uploaded_file())
    results.append(await test_openai_responses_api_with_uploaded_file())
    results.append(await test_anthropic_llm_call_with_uploaded_file())
    results.append(await test_google_llm_call_with_uploaded_file())

    print("\nRunning file deletion tests...\n")
    results.append(await test_delete_openai_file())
    results.append(await test_delete_anthropic_file())
    results.append(await test_delete_google_file())

    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("All tests passed!")
        return True
    else:
        print(f"{total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
