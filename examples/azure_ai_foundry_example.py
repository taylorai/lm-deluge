"""
Example usage of Azure AI Foundry models with lm-deluge.

Before running this example, ensure you have set the following environment variables:
- AZURE_AI_FOUNDRY_ENDPOINT: Your Azure AI Foundry deployment endpoint
  (e.g., "https://your-resource.inference.ai.azure.com")
- AZURE_AI_FOUNDRY_API_KEY: Your Azure AI Foundry API key

You can set these in your .env file or export them in your shell:
    export AZURE_AI_FOUNDRY_ENDPOINT="https://your-resource.inference.ai.azure.com"
    export AZURE_AI_FOUNDRY_API_KEY="your-api-key"
"""

import asyncio
from lm_deluge import LLMClient, Conversation


async def main():
    # Example 1: Simple chat completion with GPT-4o on Azure
    print("Example 1: Simple chat with GPT-4o on Azure")
    client = LLMClient("gpt-4o-azure", max_new_tokens=100)
    conv = Conversation().user("What are the main benefits of using Azure AI Foundry?")
    response = await client.start(conv)
    print(f"Response: {response.completion}\n")

    # Example 2: Using multiple Azure models
    print("Example 2: Distributing requests across Azure models")
    client = LLMClient(
        ["gpt-4o-mini-azure", "llama-3.1-8b-azure", "mistral-small-azure"],
        max_requests_per_minute=10,
    )
    prompts = [
        "What is machine learning?",
        "Explain neural networks in one sentence.",
        "What is the capital of France?",
    ]
    responses = await client.process_prompts_async(prompts)
    for i, resp in enumerate(responses):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Response: {resp.completion}\n")

    # Example 3: Using vision models with images
    print("Example 3: Vision model with image")
    from lm_deluge import Message

    client = LLMClient("llama-3.2-90b-azure", max_new_tokens=200)
    conv = (
        Conversation()
        .system("You are a helpful AI assistant that can analyze images.")
        .add(
            Message.user("What's in this image?").add_image(
                "tests/image.jpg"  # Path to your image
            )
        )
    )
    response = await client.start(conv)
    print(f"Image description: {response.completion}\n")

    # Example 4: JSON mode for structured outputs
    print("Example 4: JSON mode with structured output")
    from lm_deluge import SamplingParams

    client = LLMClient(
        "gpt-4o-mini-azure",
        sampling_params=SamplingParams(json_mode=True, max_new_tokens=150),
    )
    conv = Conversation().user(
        'Provide information about Paris in JSON format with keys: "city", "country", "population", "famous_landmarks"'
    )
    response = await client.start(conv)
    print(f"JSON response: {response.completion}\n")

    # Example 5: Using tools with Azure models
    print("Example 5: Tool use with Azure models")
    from lm_deluge import Tool

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 72°F"

    weather_tool = Tool.from_function(get_weather)

    client = LLMClient("gpt-4o-azure", max_new_tokens=100)
    conv = Conversation().user("What's the weather in San Francisco?")

    final_conv, response = await client.run_agent_loop(
        conv, tools=[weather_tool], max_rounds=3
    )
    print(f"Tool use response: {response.completion}\n")

    # Example 6: Find all available Azure models
    print("Example 6: Finding available Azure AI Foundry models")
    from lm_deluge.models import find_models

    azure_models = find_models(provider="azure_ai_foundry")
    print(f"Found {len(azure_models)} Azure AI Foundry models:")
    for model in azure_models[:5]:  # Show first 5
        print(
            f"  - {model.id}: ${model.input_cost}/M input tokens, ${model.output_cost}/M output tokens"
        )
    print("...")

    # Example 7: Find cheapest Azure models
    print("\nExample 7: Finding cheapest Azure models")
    cheap_azure = find_models(
        provider="azure_ai_foundry", sort_by="input_cost", limit=3
    )
    print("Top 3 cheapest Azure models by input cost:")
    for model in cheap_azure:
        print(
            f"  - {model.id}: ${model.input_cost}/M input tokens, ${model.output_cost}/M output tokens"
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nError: {e}")
        print(
            "\nMake sure you have set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_API_KEY"
        )
        print("environment variables before running this example.")
