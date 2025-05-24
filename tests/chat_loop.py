import asyncio
from lm_deluge import LLMClient, Conversation, Message


async def chat_loop():
    # Initialize the conversation with a system message
    conversation = Conversation.system("You are a helpful assistant.")

    # Create a client with your preferred model
    client = LLMClient.basic("llama-4-maverick")  # or any other supported model

    print("Chat started! Type 'quit' to exit.")
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
            # Send conversation to model
            responses = await client.process_prompts_async(
                [conversation], show_progress=False
            )

            # Extract the response
            response = responses[0]
            if response and response.completion:
                assistant_response = response.completion
                print(f"\nAssistant: {assistant_response}")

                # Add assistant response to conversation to maintain history
                conversation.add(Message.ai(assistant_response))
            else:
                print("Assistant: Sorry, I didn't get a response.")

        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(chat_loop())
