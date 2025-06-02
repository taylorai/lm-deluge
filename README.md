# lm-deluge

`lm-deluge` is a lightweight helper library for maxing out your rate limits with LLM providers. It provides the following:

- **Unified client** – Send prompts to all relevant models with a single client.
- **Files and Images** - Include images easily for multimodal models, and PDF files for models that support them (OpenAI and Anthropic).
- **Massive concurrency with throttling** – Set `max_tokens_per_minute` and `max_requests_per_minute` and let it fly. The client will process as many requests as possible while respecting rate limits and retrying failures.
- **Spray across models/providers** – Configure a client with multiple models from any provider(s), and sampling weights. The client samples a model for each request.
- **Tool Use** – Unified API for defining tools for all providers, and creating tools automatically from python functions.
- **MCP Support** – Instantiate a `Tool` from a local or remote MCP server so that any LLM can use it, whether or not that provider natively supports MCP.
- **Computer Use** – We support Claude Computer Use via the computer_use argument to process_prompts_sync/async. It works with Anthropic's API; Bedrock's API is broken right now and rejects the tool definitions, but in principle this will work there too when Bedrock gets their sh*t together.
- **Caching** – Save completions in a local or distributed cache to avoid repeated LLM calls to process the same input.
- **Convenient message constructor** – No more looking up how to build an Anthropic messages list with images. Our `Conversation` and `Message` classes work great with our client or with the `openai` and `anthropic` packages.
- **Sync and async APIs** – Use the client from sync or async code.

**STREAMING IS NOT IN SCOPE.** There are plenty of packages that let you stream chat completions across providers. The sole purpose of this package is to do very fast batch inference using APIs. Sorry!

**Update 06/02/2025:** I lied, it supports (very basic) streaming now via client.stream(...). It will print tokens as they arrive, then return an APIResponse at the end. More sophisticated streaming may or may not be implemented later, don't count on it.

## Installation

```bash
pip install lm-deluge
```

The package relies on environment variables for API keys. Typical variables include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `META_API_KEY`, and `GOOGLE_API_KEY`. `LLMClient` will automatically load the `.env` file when imported; we recommend using that to set the environment variables. For Bedrock, you'll need to set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

## Quickstart

The easiest way to get started is with the `.basic` constructor. This uses sensible default arguments for rate limits and sampling parameters so that you don't have to provide a ton of arguments.

```python
from lm_deluge import LLMClient

client = LLMClient.basic("gpt-4o-mini")
resps = client.process_prompts_sync(["Hello, world!"])
print(resp[0].completion)
```

## Spraying Across Models

To distribute your requests across models, just provide a list of more than one model to the constructor. See all available models in `models.py`. The rate limits for the client apply to the client as a whole, not per-model, so you may want to increase them:

```python
from lm_deluge import LLMClient

client = LLMClient.basic(
    ["gpt-4o-mini", "claude-3-haiku"],
    max_requests_per_minute=10_000
)
resps = client.process_prompts_sync(
    ["Hello, ChatGPT!", "Hello, Claude!"]
)
print(resp[0].completion)
```

## Configuration

API calls can be customized in a few ways.

1. **Sampling Parameters.** This determines things like structured outputs, maximum completion tokens, nucleus sampling, etc. Provide a custom `SamplingParams` to the `LLMClient` to set temperature, top_p, json_mode, max_new_tokens, and/or reasoning_effort. You can pass 1 `SamplingParams` to use for all models, or a list of `SamplingParams` that's the same length as the list of models. You can also pass many of these arguments directly to `LLMClient.basic` so you don't have to construct an entire `SamplingParams` object.
2. **Arguments to LLMClient.** This is where you set request timeout, rate limits, model name(s), model weight(s) for distributing requests across models, retries, and caching.
3. **Arguments to process_prompts.** Per-call, you can set verbosity, whether to display progress, and whether to return just completions (rather than the full APIResponse object). This is also where you provide tools.

Putting it all together:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    "gpt-4",
    max_requests_per_minute=100,
    max_tokens_per_minute=100_000,
    max_concurrent_requests=500,
    sampling_params=SamplingParams(temperature=0.5, max_new_tokens=30)
)

await client.process_prompts_async(
    ["What is the capital of Mars?"],
    show_progress=False,
    return_completions_only=True
)
```

## Multi-Turn Conversations

Constructing conversations to pass to models is notoriously annoying. Each provider has a slightly different way of defining a list of messages, and with the introduction of images/multi-part messages it's only gotten worse. We provide convenience constructors so you don't have to remember all that stuff.

```python
from lm_deluge import Message, Conversation

prompt = Conversation.system("You are a helpful assistant.").add(
    Message.user("What's in this image?").add_image("tests/image.jpg")
)

client = LLMClient.basic("gpt-4.1-mini")
resps = client.process_prompts_sync([prompt])
```

This just works. Images can be local images on disk, URLs, bytes, base64 data URLs... go wild. You can use `Conversation.to_openai` or `Conversation.to_anthropic` to format your messages for the OpenAI or Anthropic clients directly.

See a full multi-turn chat example in `examples/multiturn.md`.

## Tool Use

Define tools from Python functions and use them with any model:

```python
from lm_deluge import LLMClient, Tool

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 72°F"

tool = Tool.from_function(get_weather)
client = LLMClient.basic("claude-3-haiku")
resps = client.process_prompts_sync(
    ["What's the weather in Paris?"],
    tools=[tool]
)

# you can iterate over the tool calls in the response automatically
for tool_call in resps[0].tool_calls:
    print(tool_call.name, tool_call.arguments)
```

You can also automatically instantiate tools from MCP servers. Under the hood, the the constructor connects to the server, asks it what tools it has, and then creates a `Tool` from each of them, *with a built-in `call` and `acall` interface*.

```python
from lm_deluge import LLMClient, Tool

# Connect to a local MCP server and get all of its tools
filesystem_tools = Tool.from_mcp(
    "filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)

# or load ALL the tools from a Claude Desktop like config
config = {
    "mcpServers": {
        "exa": {
            "url": f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}"
        },
        "zapier": {
            "url": f"https://mcp.zapier.com/api/mcp/s/{os.getenv('ZAPIER_MCP_SECRET')}/mcp"
        }
    }
}
all_tools = Tool.from_mcp_config(config)

# let the model use the tools
client = LLMClient.basic("gpt-4o-mini")
resps = client.process_prompts_sync(
    ["List the files in the current directory"],
    tools=tools
)

# call the tools
for tool_call in resps[0].tool_calls:
    # this is dumb sorry will make it better
    tool_to_call = [x for x in tools if x.name == tool_call.name][0]
    tool_to_call.call(**tool_call.arguments) # in async code, use .acall()
```

### Prompt Caching (Anthropic)

For Anthropic models, you can use prompt caching to reduce costs and latency for repeated context. This uses Anthropic's server-side prompt caching. Other providers like OpenAI and Google do this automatically, but Anthropic requires you to manually set cache-control on messages. You can do this in lm-deluge with a simple "cache" argument to `process_prompts_sync` or `process_prompts_async`:

```python
from lm_deluge import LLMClient, Conversation, Message

# Create a conversation with system message
conv = (
    Conversation.system("You are an expert Python developer with deep knowledge of async programming.")
    .add(Message.user("How do I use asyncio.gather?"))
)

# Use prompt caching to cache system message and tools
client = LLMClient.basic("claude-3-5-sonnet")
resps = client.process_prompts_sync(
    [conv],
    cache="system_and_tools"  # Cache system message and any tools
)
```

Available cache patterns: `"system_and_tools"`, `"tools_only"`, `"last_user_message"`, `"last_2_user_messages"`, `"last_3_user_messages"`.

## Local Caching

Besides caching from model providers (which provides cache reads at a discount, but not for free) `lm_deluge.cache` includes LevelDB, SQLite and custom dictionary based caches to cache prompts locally. Pass an instance via `LLMClient(..., cache=my_cache)` and previously seen prompts will not be re‑sent across different `process_prompts_[...]` calls.

**IMPORTANT:** Caching does not currently work for prompts in the SAME batch. That is, if you call `process_prompts_sync` with the same prompt 100 times, there will be 0 cache hits. If you call `process_prompts_sync` a *second* time with those same 100 prompts, all 100 will be cache hits. The local cache is intended to be persistent and help you save costs across many invocations, but it can't help with a single batch-inference session (yet!).

## Asynchronous Client
Use this in asynchronous code, or in a Jupyter notebook. If you try to use the sync client in a Jupyter notebook, you'll have to use `nest-asyncio`, because internally the sync client uses async code. Don't do it! Just use the async client!

```python
import asyncio

async def main():
    responses = await client.process_prompts_async(
        ["an async call"],
        return_completions_only=True,
    )
    print(responses[0])

asyncio.run(main())
```

## Available Models

We support all models in `src/lm_deluge/models.py`. Vertex support is not planned in the short term, since Google allows you to connect your Vertex account to AI Studio, and Vertex authentication is a huge pain (requires service account credentials, etc.)

## Feature Support

We support structured outputs via `json_mode` parameter provided to `SamplingParams`. Structured outputs with a schema are planned. Reasoning models are supported via the `reasoning_effort` parameter, which is translated to a thinking budget for Claude/Gemini. Image models are supported. We support tool use as documented above. We support logprobs for OpenAI models that return them.

## Built‑in tools

The `lm_deluge.llm_tools` package exposes a few helper functions:

- `extract` – structure text or images into a Pydantic model based on a schema.
- `translate` – translate a list of strings to English.
- `score_llm` – simple yes/no style scoring with optional log probability output.

Experimental embeddings (`embed.embed_parallel_async`) and document reranking (`rerank.rerank_parallel_async`) clients are also provided.
