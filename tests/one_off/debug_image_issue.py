# #!/usr/bin/env python3

# import asyncio
# import base64

# import dotenv

# from lm_deluge import LLMClient
# from lm_deluge.prompt import Conversation, Message

# dotenv.load_dotenv()

# # Create a small fake JPEG image in base64
# # This is a minimal valid JPEG header + data
# fake_jpeg_b64 = "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAACAAIDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="

# print("=== DEBUG: Starting image processing test ===")


# async def test_image_issue():
#     # Mimic your PDF processing code
#     print("1. Creating client with cache")
#     from lm_deluge.cache import LevelDBCache

#     cache = LevelDBCache("/tmp/debug_cache")
#     client = LLMClient(
#         "gemini-2.5-flash-lite", request_timeout=180, cache=cache
#     )  # Re-enable cache

#     print("2. Creating system message")
#     system_prompt = Message.system(
#         "You are a document intelligence model. You ALWAYS answer in English, regardless of the language of the input document."
#     )

#     print("3. Creating user message with image")
#     doc_prompt = Message.user("Summarize this document")

#     # Decode base64 to bytes (this is what your code does)
#     image_bytes = base64.b64decode(fake_jpeg_b64)
#     print(f"4. Adding image - bytes length: {len(image_bytes)}")

#     # This should trigger Image.__post_init__ debug log
#     doc_prompt.add_image(image_bytes, media_type="image/jpeg")

#     print("5. Creating conversation")
#     conversation = Conversation([system_prompt, doc_prompt])

#     print("6. Processing prompt (first time)...")
#     try:
#         # This should trigger various debug logs
#         resps = await client.process_prompts_async([conversation])
#         print(f"7. Got response: {resps[0].completion[:100]}...")

#         print("\n8. Processing same prompt again (should hit cache)...")
#         resps2 = await client.process_prompts_async([conversation])
#         print(f"9. Got cached response: {resps2[0].completion[:100]}...")

#     except Exception as e:
#         print(f"ERROR: {e}")
#         print("This is the error we're debugging!")


# if __name__ == "__main__":
#     asyncio.run(test_image_issue())
