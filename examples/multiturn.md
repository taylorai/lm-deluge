# Multi-turn Conversations with LM Deluge

This guide demonstrates how to create interactive chat loops using LM Deluge's conversation system. The library makes it easy to maintain conversation state across multiple turns by simply adding new messages to the `Conversation` object.

## Simple Chat Loop

Here's a basic example of an interactive chat loop that maintains conversation history:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def chat_loop():
    # Initialize the conversation with a system message
    conversation = Conversation().system("You are a helpful assistant.")

    # Create a client with your preferred model
    client = LLMClient("gpt-4o-mini")  # or any other supported model

    print("Chat started! Type 'quit' to exit.")
    print("Assistant: Hello! How can I help you today?")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Exit condition
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Assistant: Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to conversation
        conversation.add(Message.user(user_input))

        try:
            # Send conversation to model
            responses = await client.process_prompts_async(
                [conversation],
                show_progress=False
            )

            # Extract the response
            response = responses[0]
            if response and response.completion:
                assistant_response = response.completion
                print(f"Assistant: {assistant_response}")

                # Add assistant response to conversation to maintain history
                conversation.add(Message.ai(assistant_response))
            else:
                print("Assistant: Sorry, I didn't get a response.")

        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    asyncio.run(chat_loop())
```

## Enhanced Chat Loop with Error Handling

Here's a more robust version with better error handling and conversation management:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def enhanced_chat_loop():
    # Initialize conversation
    conversation = Conversation().system(
        "You are a helpful assistant. Keep your responses concise and helpful."
    )

    # Initialize client with error handling
    try:
        client = LLMClient(
            "gpt-4o-mini",
            max_new_tokens=500,  # Limit response length
            temperature=0.7,
            max_requests_per_minute=60
        )
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    print("🤖 Enhanced Chat Assistant")
    print("=" * 40)
    print("Commands:")
    print("  'quit' or 'exit' - End the conversation")
    print("  'history' - Show conversation history")
    print("  'clear' - Clear conversation history")
    print("=" * 40)
    print("Assistant: Hello! I'm ready to help. What would you like to know?")

    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()

            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("🤖 Assistant: Thanks for chatting! Goodbye!")
                break

            elif user_input.lower() == 'history':
                print("\n📝 Conversation History:")
                for i, msg in enumerate(conversation.messages, 1):
                    role = msg.role.title()
                    text = msg.completion or "[No text]"
                    print(f"  {i}. {role}: {text[:100]}{'...' if len(text) > 100 else ''}")
                continue

            elif user_input.lower() == 'clear':
                conversation = Conversation().system(
                    "You are a helpful assistant. Keep your responses concise and helpful."
                )
                print("🗑️  Conversation history cleared!")
                continue

            if not user_input:
                continue

            # Add user message
            conversation.add(Message.user(user_input))

            # Show thinking indicator
            print("🤖 Assistant: Thinking...")

            # Get response
            responses = await client.process_prompts_async(
                [conversation],
                show_progress=False
            )

            response = responses[0]
            if response and response.completion:
                assistant_response = response.completion.strip()
                print(f"🤖 Assistant: {assistant_response}")

                # Add to conversation history
                conversation.add(Message.ai(assistant_response))

            else:
                print("🤖 Assistant: I'm sorry, I couldn't process that request.")
                # Remove the user message since we couldn't respond
                conversation.messages.pop()

        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")
            # Remove the last user message if there was an error
            if conversation.messages and conversation.messages[-1].role == "user":
                conversation.messages.pop()

if __name__ == "__main__":
    asyncio.run(enhanced_chat_loop())
```

## Key Features

### Conversation State Management
- The `Conversation` object automatically maintains the full conversation history
- Simply add new messages with `conversation.add(Message.user(text))` and `conversation.add(Message.ai(text))`
- The model receives the entire conversation context on each request

### Supported Message Types
- `Message.user(text)` - User messages
- `Message.ai(text)` - Assistant messages
- `Message.system(text)` - System messages (typically used once at the beginning)

### Model Compatibility
- Works with any model supported by LM Deluge
- Examples: `"gpt-4o"`, `"claude-4-sonnet"`, `"llama-3.3-70b"`, `"gemini-2.0-flash"`
- Simply change the model name in `LLMClient(model_name)`

### Tips for Multi-turn Conversations

1. **System Messages**: Set up the assistant's behavior with a system message at the start
2. **Conversation Limits**: Be aware that very long conversations may hit token limits
3. **Error Handling**: Always handle potential API errors gracefully
4. **State Persistence**: The conversation object maintains all history automatically
5. **Response Processing**: Check that responses exist before using them

## Running the Examples

Save either example to a Python file and run:

```bash
python chat_example.py
```

Make sure you have your API keys configured as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) depending on which model you're using.
