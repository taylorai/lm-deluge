import asyncio
from lm_deluge import LLMClient, Conversation, Message


def print_usage_info(usage):
    """Print usage information in a pretty, readable format."""
    if not usage:
        print("📊 Usage: No usage data available")
        return

    print("📊 Usage Information:")
    print(f"   📥 Input tokens: {usage.input_tokens:,}")
    print(f"   📤 Output tokens: {usage.output_tokens:,}")
    print(f"   📦 Total tokens: {usage.total_tokens:,}")

    if usage.cache_read_tokens or usage.cache_write_tokens:
        print("   🔄 Cache Activity:")
        if usage.cache_write_tokens and usage.cache_write_tokens > 0:
            print(f"      ✏️  Cache write: {usage.cache_write_tokens:,} tokens")
        if usage.cache_read_tokens and usage.cache_read_tokens > 0:
            print(f"      ✅ Cache read: {usage.cache_read_tokens:,} tokens")
            if usage.input_tokens > 0:
                cache_ratio = (usage.cache_read_tokens / usage.total_input_tokens) * 100
                print(f"      📈 Cache hit ratio: {cache_ratio:.1f}%")
    else:
        print("   💾 Cache: No cache activity")
    print()


async def chat_loop():
    # Initialize the conversation with a system message
    conversation = Conversation().system(
        "You are a helpful AI assistant. Provide clear, concise, and accurate responses. "
        "When appropriate, ask follow-up questions to better understand the user's needs. "
        "Be friendly and professional in your interactions."
    )

    # Create a client with Claude 3 Haiku for caching support
    client = LLMClient("claude-3-haiku", max_new_tokens=1000)

    print("Chat started with Claude 3 Haiku and prompt caching enabled!")
    print("Cache strategy: last_2_user_messages (caches system + conversation history)")
    print("Type 'quit' to exit.")
    print("=" * 70)
    print("Assistant: Hello! How can I help you today?")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Exit condition
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Assistant: Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to conversation
        conversation.add(Message.user(user_input))

        try:
            # Send conversation to model with caching enabled
            responses = await client.process_prompts_async(
                [conversation], show_progress=False, cache="last_2_user_messages"
            )

            # Extract the response
            response = responses[0]
            if response and response.completion:
                # Print usage information first
                print_usage_info(response.usage)

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
