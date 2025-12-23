"""
Integration tests for the proxy server using FastAPI TestClient.
"""

import os


def test_health_endpoint():
    """Test health check endpoint."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health endpoint: OK")


def test_models_endpoint():
    """Test /v1/models endpoint."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    app = create_app()
    client = TestClient(app)

    # Test with ?all=true to get all models regardless of API keys
    response = client.get("/v1/models?all=true")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    # Check that known models are in the list
    model_ids = [m["id"] for m in data["data"]]
    assert "gpt-4.1" in model_ids
    assert "claude-4-sonnet" in model_ids
    print(f"Models endpoint (all): OK ({len(model_ids)} models available)")

    # Test default behavior (only models with API keys set)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    # This may be empty if no API keys are set, which is fine
    available_count = len(data["data"])
    print(f"Models endpoint (available): OK ({available_count} models with API keys)")


def test_auth_when_disabled():
    """Test that auth is not required when DELUGE_PROXY_API_KEY is not set."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    # Ensure no API key is set
    os.environ.pop("DELUGE_PROXY_API_KEY", None)

    app = create_app()
    client = TestClient(app)

    # Should work without auth
    response = client.get("/v1/models")
    assert response.status_code == 200
    print("Auth disabled: OK")


def test_auth_when_enabled():
    """Test that auth is required when DELUGE_PROXY_API_KEY is set."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    # Set API key
    os.environ["DELUGE_PROXY_API_KEY"] = "test-secret-key"

    try:
        app = create_app()
        client = TestClient(app)

        # Should fail without auth
        response = client.get("/v1/models")
        assert response.status_code == 401
        print("Auth required (no header): OK")

        # Should fail with wrong auth
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401
        print("Auth required (wrong key): OK")

        # Should succeed with correct auth
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer test-secret-key"}
        )
        assert response.status_code == 200
        print("Auth required (correct key): OK")

    finally:
        os.environ.pop("DELUGE_PROXY_API_KEY", None)


def test_anthropic_auth_header():
    """Test Anthropic-style x-api-key header."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    os.environ["DELUGE_PROXY_API_KEY"] = "test-secret-key"

    try:
        app = create_app()
        client = TestClient(app)

        # Should fail without auth
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-4-sonnet",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 401

        # Should succeed with x-api-key header (auth passes, but may fail at provider)
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-secret-key"},
            json={
                "model": "claude-4-sonnet",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Auth passed if we get a response (even if provider fails due to missing key)
        # Check that our auth layer didn't reject it by looking at the error format
        data = response.json()
        # If we get an Anthropic-style error with authentication_error from the backend,
        # that means our proxy auth passed and the request was forwarded
        if response.status_code == 401:
            # Check if it's a backend auth error (means our proxy auth passed)
            assert "authentication_error" in str(data) or "api_error" in str(data)
        print("Anthropic x-api-key header: OK")

    finally:
        os.environ.pop("DELUGE_PROXY_API_KEY", None)


def test_streaming_rejected():
    """Test that streaming requests are rejected."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    os.environ.pop("DELUGE_PROXY_API_KEY", None)

    app = create_app()
    client = TestClient(app)

    # OpenAI streaming
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 400
    assert "stream" in response.json()["detail"].lower()
    print("OpenAI streaming rejected: OK")

    # Anthropic streaming
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-4-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 400
    assert "stream" in response.json()["detail"].lower()
    print("Anthropic streaming rejected: OK")


def test_invalid_model():
    """Test that invalid model names are rejected."""
    from fastapi.testclient import TestClient

    from lm_deluge.server import create_app

    os.environ.pop("DELUGE_PROXY_API_KEY", None)

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent-model-12345",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()
    print("Invalid model rejected: OK")


def test_cache_pattern_env_var():
    """Test that DELUGE_CACHE_PATTERN env var is read correctly."""
    from lm_deluge.server.app import get_cache_pattern

    # Test unset (default)
    os.environ.pop("DELUGE_CACHE_PATTERN", None)
    assert get_cache_pattern() is None
    print("Cache pattern (unset): OK")

    # Test 'none'
    os.environ["DELUGE_CACHE_PATTERN"] = "none"
    assert get_cache_pattern() is None
    print("Cache pattern (none): OK")

    # Test 'NONE' (case insensitive)
    os.environ["DELUGE_CACHE_PATTERN"] = "NONE"
    assert get_cache_pattern() is None
    print("Cache pattern (NONE): OK")

    # Test valid patterns
    valid_patterns = [
        "tools_only",
        "system_and_tools",
        "last_user_message",
        "last_2_user_messages",
        "last_3_user_messages",
    ]
    for pattern in valid_patterns:
        os.environ["DELUGE_CACHE_PATTERN"] = pattern
        assert get_cache_pattern() == pattern
        print(f"Cache pattern ({pattern}): OK")

    # Test case insensitive
    os.environ["DELUGE_CACHE_PATTERN"] = "TOOLS_ONLY"
    assert get_cache_pattern() == "tools_only"
    print("Cache pattern (case insensitive): OK")

    # Test invalid pattern
    os.environ["DELUGE_CACHE_PATTERN"] = "invalid_pattern"
    assert get_cache_pattern() is None
    print("Cache pattern (invalid): OK")

    # Cleanup
    os.environ.pop("DELUGE_CACHE_PATTERN", None)


if __name__ == "__main__":
    test_health_endpoint()
    test_models_endpoint()
    test_auth_when_disabled()
    test_auth_when_enabled()
    test_anthropic_auth_header()
    test_streaming_rejected()
    test_invalid_model()
    test_cache_pattern_env_var()
    print("\nAll integration tests passed!")
