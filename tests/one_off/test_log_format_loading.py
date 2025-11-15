#!/usr/bin/env python3
"""Manual test to demonstrate from_unknown() handling log format."""

from lm_deluge.prompt import Conversation, Message

# Create a conversation

convo = Conversation()
convo.add(Message.system("You are a helpful assistant."))
convo.add(Message.user("Hello, how are you?"))
convo.add(Message.ai("I'm doing great, thanks for asking!"))

print("Original conversation:")
print(f"  Messages: {len(convo.messages)}")
for i, msg in enumerate(convo.messages):
    print(f"  {i+1}. {msg.role}: {msg.parts[0].text if msg.parts else ''}")

# Convert to log format
log_data = convo.to_log()
print("\nLog format structure:")
print(f"  Keys: {list(log_data.keys())}")
print(f"  Number of messages: {len(log_data['messages'])}")

# Test 1: Load from log format using from_unknown()
print("\n--- Test 1: Load from log format ---")
loaded_convo, provider = Conversation.from_unknown(log_data)
print(f"  Detected provider: {provider}")
print(f"  Messages loaded: {len(loaded_convo.messages)}")
for i, msg in enumerate(loaded_convo.messages):
    print(f"  {i+1}. {msg.role}: {msg.parts[0].text if msg.parts else ''}")

# Test 2: Load from OpenAI format
print("\n--- Test 2: Load from OpenAI format ---")
openai_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]
openai_convo, provider = Conversation.from_unknown(openai_messages)
print(f"  Detected provider: {provider}")
print(f"  Messages loaded: {len(openai_convo.messages)}")

# Test 3: Load from Anthropic format
print("\n--- Test 3: Load from Anthropic format ---")
anthropic_messages = [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
]
anthropic_convo, provider = Conversation.from_unknown(anthropic_messages)
print(f"  Detected provider: {provider}")
print(f"  Messages loaded: {len(anthropic_convo.messages)}")

print("\n✅ All tests passed!")
