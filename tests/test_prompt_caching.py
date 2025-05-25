import json
from lm_deluge.prompt import Conversation, Message, CachePattern
from lm_deluge.tool import Tool


def test_cache_patterns():
    """Test that cache control is correctly applied for different patterns."""
    # Create a conversation with system message and user message
    conv = (
        Conversation.system("You are a helpful assistant.")
        .add(Message.user("Hello, how are you?"))
        .add(Message.ai("I'm doing well, thank you!"))
        .add(Message.user("What's the weather like?"))
    )
    
    # Test system_and_tools caching
    system_msg, messages = conv.to_anthropic(cache_pattern="system_and_tools")
    assert isinstance(system_msg, list), "System message should be structured format for caching"
    assert system_msg[0]["cache_control"] == {"type": "ephemeral"}, "System should have cache control"
    
    # Test last_user_message caching
    system_msg, messages = conv.to_anthropic(cache_pattern="last_user_message")
    last_user_idx = None
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            last_user_idx = i
    
    assert last_user_idx is not None, "Should have user messages"
    content = messages[last_user_idx]["content"]
    if isinstance(content, list):
        assert content[-1]["cache_control"] == {"type": "ephemeral"}, "Last user message should have cache control"
    else:
        # Should be converted to structured format
        assert False, "User message content should be converted to list format for caching"


def test_tools_only_caching():
    """Test that tools_only caching works at the API request level."""
    from lm_deluge.api_requests.anthropic import AnthropicRequest
    from lm_deluge.tracker import StatusTracker
    import asyncio
    
    # Create a simple conversation
    conv = Conversation.user("Hello")
    
    # Create a simple tool
    def test_function(arg: str) -> str:
        """A test function."""
        return f"Result: {arg}"
    
    tool = Tool.from_function(test_function)
    
    # Create request with tools_only caching
    status_tracker = StatusTracker()
    retry_queue = asyncio.Queue()
    
    request = AnthropicRequest(
        task_id=1,
        model_name="claude-3-5-sonnet-20241022",
        prompt=conv,
        attempts_left=1,
        status_tracker=status_tracker,
        retry_queue=retry_queue,
        results_arr=[],
        tools=[tool],
        cache="tools_only",
        all_model_names=["claude-3-5-sonnet-20241022"],
    )
    
    # Check that cache control was added to the last tool
    tools = request.request_json.get("tools", [])
    assert len(tools) > 0, "Should have tools"
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}, "Last tool should have cache control"


def test_image_locking():
    """Test that images are locked as bytes when caching is enabled."""
    from lm_deluge.image import Image
    import io
    
    # Create a conversation with an image
    conv = Conversation.user("What's in this image?")
    # Add a simple 1x1 PNG image as bytes
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xddã\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    conv.messages[0].add_image(png_bytes)
    
    # Lock images as bytes
    conv.lock_images_as_bytes()
    
    # Check that the image data is now bytes
    image_parts = conv.messages[0].images
    assert len(image_parts) > 0, "Should have image parts"
    assert isinstance(image_parts[0].data, bytes), "Image data should be bytes after locking"


def test_no_cache_control_without_cache():
    """Test that no cache control is added when cache is None."""
    conv = Conversation.system("You are helpful.").add(Message.user("Hello"))
    
    system_msg, messages = conv.to_anthropic(cache_pattern=None)
    
    # System should be a string, not structured
    assert isinstance(system_msg, str), "System should remain as string without caching"
    
    # Messages should not have cache control
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block, "Should not have cache control without caching"


if __name__ == "__main__":
    test_cache_patterns()
    test_tools_only_caching()
    test_image_locking()
    test_no_cache_control_without_cache()
    print("All prompt caching tests passed!")