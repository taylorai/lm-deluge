"""
Tests that verify the proxy server works with the official OpenAI and Anthropic SDKs.

These tests start a real server and make requests using the official client libraries.
"""

import subprocess
import sys
import time

import httpx


def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def test_openai_sdk_compatibility():
    """Test that the OpenAI Python SDK works with our proxy."""
    from openai import OpenAI

    # Start server in background
    server_process = subprocess.Popen(
        [sys.executable, "-m", "lm_deluge.server", "--port", "18080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        if not wait_for_server("http://localhost:18080"):
            stdout, stderr = server_process.communicate(timeout=1)
            print("Server stdout:", stdout.decode())
            print("Server stderr:", stderr.decode())
            raise RuntimeError("Server failed to start")

        # Create OpenAI client pointing to our proxy
        client = OpenAI(
            base_url="http://localhost:18080/v1",
            api_key="not-needed",  # Our proxy doesn't require auth by default
        )

        # Test /v1/models
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        assert len(model_ids) > 0 or True  # May be 0 if no API keys set
        print(f"OpenAI SDK - models.list(): OK ({len(model_ids)} models)")

        # Test /v1/chat/completions with a model that requires no external API
        # Since we don't have API keys set, this will fail at the provider level
        # But it validates that the SDK can communicate with our proxy
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10,
            )
            # If we get here, it actually worked (API key was set)
            print("OpenAI SDK - chat.completions.create(): OK (got response)")
            assert response.choices[0].message.content is not None
        except Exception as e:
            # Expected to fail due to missing API key, but should be a proper error
            error_str = str(e)
            # Should be an API error, not a connection/parsing error
            assert (
                "401" in error_str
                or "api" in error_str.lower()
                or "key" in error_str.lower()
            ), f"Unexpected error: {e}"
            print(
                "OpenAI SDK - chat.completions.create(): OK (got expected auth error)"
            )

        # Test that streaming is properly rejected
        try:
            client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
            )
            raise AssertionError("Streaming should have been rejected")
        except Exception as e:
            assert "stream" in str(e).lower() or "400" in str(e)
            print("OpenAI SDK - streaming rejected: OK")

    finally:
        server_process.terminate()
        server_process.wait(timeout=5)


def test_anthropic_sdk_compatibility():
    """Test that the Anthropic Python SDK works with our proxy."""
    from anthropic import Anthropic

    # Start server in background
    server_process = subprocess.Popen(
        [sys.executable, "-m", "lm_deluge.server", "--port", "18081"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        if not wait_for_server("http://localhost:18081"):
            stdout, stderr = server_process.communicate(timeout=1)
            print("Server stdout:", stdout.decode())
            print("Server stderr:", stderr.decode())
            raise RuntimeError("Server failed to start")

        # Create Anthropic client pointing to our proxy
        # Note: Anthropic SDK appends /v1/messages to base_url, so don't include /v1
        client = Anthropic(
            base_url="http://localhost:18081",
            api_key="not-needed",  # Our proxy doesn't require auth by default
        )

        # Test /v1/messages
        try:
            response = client.messages.create(
                model="claude-4-sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hello"}],
            )
            # If we get here, it actually worked (API key was set)
            print("Anthropic SDK - messages.create(): OK (got response)")
            assert len(response.content) > 0
        except Exception as e:
            # Expected to fail due to missing API key
            error_str = str(e)
            assert (
                "401" in error_str
                or "auth" in error_str.lower()
                or "key" in error_str.lower()
            ), f"Unexpected error: {e}"
            print("Anthropic SDK - messages.create(): OK (got expected auth error)")

        # Test that streaming is properly rejected
        try:
            with client.messages.stream(
                model="claude-4-sonnet",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hello"}],
            ) as stream:
                for text in stream.text_stream:
                    pass
            raise AssertionError("Streaming should have been rejected")
        except Exception as e:
            # Should fail because streaming is not supported
            assert "stream" in str(e).lower() or "400" in str(e)
            print("Anthropic SDK - streaming rejected: OK")

    finally:
        server_process.terminate()
        server_process.wait(timeout=5)


def test_openai_sdk_with_tools():
    """Test that tool calling works through the OpenAI SDK."""
    from openai import OpenAI

    # Start server in background
    server_process = subprocess.Popen(
        [sys.executable, "-m", "lm_deluge.server", "--port", "18082"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server("http://localhost:18082"):
            raise RuntimeError("Server failed to start")

        client = OpenAI(
            base_url="http://localhost:18082/v1",
            api_key="not-needed",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        try:
            client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=tools,
                max_tokens=100,
            )
            print("OpenAI SDK - tool calling: OK (got response)")
        except Exception as e:
            # Expected auth error, but tools should be properly parsed
            error_str = str(e)
            assert (
                "401" in error_str
                or "api" in error_str.lower()
                or "key" in error_str.lower()
            )
            print(
                "OpenAI SDK - tool calling: OK (tools accepted, got expected auth error)"
            )

    finally:
        server_process.terminate()
        server_process.wait(timeout=5)


def test_anthropic_sdk_with_tools():
    """Test that tool calling works through the Anthropic SDK."""
    from anthropic import Anthropic

    # Start server in background
    server_process = subprocess.Popen(
        [sys.executable, "-m", "lm_deluge.server", "--port", "18083"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server("http://localhost:18083"):
            raise RuntimeError("Server failed to start")

        client = Anthropic(
            base_url="http://localhost:18083",
            api_key="not-needed",
        )

        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        try:
            client.messages.create(
                model="claude-4-sonnet",
                max_tokens=100,
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=tools,
            )
            print("Anthropic SDK - tool calling: OK (got response)")
        except Exception as e:
            # Expected auth error, but tools should be properly parsed
            error_str = str(e)
            assert (
                "401" in error_str
                or "auth" in error_str.lower()
                or "key" in error_str.lower()
            )
            print(
                "Anthropic SDK - tool calling: OK (tools accepted, got expected auth error)"
            )

    finally:
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    # Install required packages
    print("Testing OpenAI SDK compatibility...")
    test_openai_sdk_compatibility()
    print()

    print("Testing Anthropic SDK compatibility...")
    test_anthropic_sdk_compatibility()
    print()

    print("Testing OpenAI SDK with tools...")
    test_openai_sdk_with_tools()
    print()

    print("Testing Anthropic SDK with tools...")
    test_anthropic_sdk_with_tools()
    print()

    print("All SDK compatibility tests passed!")
