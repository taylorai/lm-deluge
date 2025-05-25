# Anthropic Prompt Caching with lm-deluge

This guide demonstrates how to use Anthropic's prompt caching feature with lm-deluge to reduce costs and latency when working with repeated context.

## What is Prompt Caching?

Anthropic's prompt caching allows you to cache parts of your prompt on their servers, so subsequent requests that reuse the same context are cheaper and faster. This is particularly useful for:

- Long system prompts that remain constant
- Large tool definitions that don't change
- Context windows that get reused across multiple queries
- Multi-turn conversations where earlier messages remain constant

## Basic Usage

```python
from lm_deluge import LLMClient, Conversation, Message

# Create a client - prompt caching works with any Anthropic model
client = LLMClient.basic("claude-3-5-sonnet")

# Create a conversation with a long system prompt
conv = Conversation.system("""
You are an expert software architect with 20+ years of experience in:
- Distributed systems design
- Microservices architecture  
- Database optimization
- Cloud infrastructure (AWS, GCP, Azure)
- Container orchestration (Kubernetes, Docker)
- CI/CD pipelines and DevOps practices
- Security best practices and compliance
- Performance optimization and monitoring

You provide detailed, practical advice based on industry best practices
and real-world experience. Your responses include concrete examples,
potential pitfalls to avoid, and step-by-step implementation guidance.
""").add(Message.user("How should I design a scalable chat application?"))

# Use system_and_tools caching to cache the system prompt
resps = client.process_prompts_sync(
    [conv],
    cache="system_and_tools"
)

print(resps[0].completion)
```

## Cache Patterns

lm-deluge supports several cache patterns:

### 1. `system_and_tools` - Cache System Message and Tools

This caches the system message and any tool definitions. Best for scenarios where you have a consistent system prompt and tool set.

```python
from lm_deluge import Tool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate tip amount."""
    return bill_amount * (tip_percentage / 100)

tools = [
    Tool.from_function(get_weather),
    Tool.from_function(calculate_tip)
]

conv = Conversation.system("You are a helpful assistant.").add(
    Message.user("What's the weather in Paris and what's a 20% tip on $50?")
)

# Cache system message and tool definitions
resps = client.process_prompts_sync(
    [conv],
    tools=tools,
    cache="system_and_tools"
)
```

### 2. `tools_only` - Cache Only Tool Definitions

Cache just the tools, not the system message. Useful when tools are complex but system prompts vary.

```python
# Multiple different system prompts but same tools
system_prompts = [
    "You are a helpful restaurant assistant.",
    "You are a travel planning expert.",
    "You are a financial advisor."
]

for system_prompt in system_prompts:
    conv = Conversation.system(system_prompt).add(
        Message.user("Help me plan something")
    )
    
    resps = client.process_prompts_sync(
        [conv],
        tools=tools,
        cache="tools_only"  # Only cache the tools
    )
```

### 3. `last_user_message` - Cache the Last User Message

Useful for iterative refinement where you're building on the same base query.

```python
# Base conversation that will be reused
base_conv = (
    Conversation.system("You are a code reviewer.")
    .add(Message.user("Please review this Python function:"))
    .add(Message.user("""
def process_data(items):
    result = []
    for item in items:
        if item.status == 'active':
            processed = transform_item(item)
            if processed:
                result.append(processed)
    return result
"""))
)

# First request - establish cache
resps = client.process_prompts_sync(
    [base_conv],
    cache="last_user_message"
)

# Follow-up requests reuse the cached code
follow_up_conv = base_conv.add(Message.user("Now focus specifically on performance optimization"))

resps2 = client.process_prompts_sync(
    [follow_up_conv],
    cache="last_user_message"  # The code block is cached
)
```

### 4. `last_2_user_messages` and `last_3_user_messages`

Cache multiple recent user messages for longer context reuse.

```python
# Long conversation with context to cache
conv = (
    Conversation.system("You are a SQL expert.")
    .add(Message.user("I have a database schema with users, orders, and products tables."))
    .add(Message.ai("Great! I can help you with SQL queries for that schema."))
    .add(Message.user("Users table has: id, name, email, created_at"))
    .add(Message.user("Orders table has: id, user_id, product_id, quantity, order_date"))
    .add(Message.user("Now write a query to find top customers by total order value"))
)

# Cache the last 2 user messages (schema definitions)
resps = client.process_prompts_sync(
    [conv],
    cache="last_2_user_messages"
)
```

## Cost Savings Example

Here's a practical example showing how caching reduces costs:

```python
import asyncio
from lm_deluge import LLMClient

async def demonstrate_cost_savings():
    client = LLMClient.basic("claude-3-5-sonnet")
    
    # Long system prompt (expensive to process repeatedly)
    long_system_prompt = "You are an expert data scientist..." * 100  # Very long prompt
    
    questions = [
        "What is linear regression?",
        "Explain decision trees",
        "How does random forest work?",
        "What is cross-validation?",
        "Explain feature engineering"
    ]
    
    # First batch - cache gets created (normal cost)
    conversations = [
        Conversation.system(long_system_prompt).add(Message.user(q))
        for q in questions
    ]
    
    print("Processing with caching...")
    resps = await client.process_prompts_async(
        conversations,
        cache="system_and_tools"
    )
    
    # Check usage to see cache savings
    for i, resp in enumerate(resps):
        usage = resp.usage
        if usage.has_cache_hit:
            print(f"Question {i+1}: Cache hit! Read {usage.cache_read_tokens} tokens from cache")
            print(f"  Input tokens: {usage.input_tokens}, Cache read: {usage.cache_read_tokens}")
        else:
            print(f"Question {i+1}: No cache hit")

asyncio.run(demonstrate_cost_savings())
```

## Multi-turn Conversations with Caching

When building multi-turn conversations, you can cache earlier parts of the conversation:

```python
# Start with a cached system message
conv = Conversation.system("""
You are an expert Python tutor. You explain concepts clearly and provide
practical examples. You adapt your teaching style to the student's level.
""")

# First interaction
conv.add(Message.user("I'm new to Python. Can you explain what a list is?"))

# Process with caching
resps = client.process_prompts_sync([conv], cache="system_and_tools")
conv.add(Message.ai(resps[0].completion))

# Continue conversation - system message is cached
conv.add(Message.user("How do I add items to a list?"))
resps2 = client.process_prompts_sync([conv], cache="system_and_tools")
conv.add(Message.ai(resps2[0].completion))

# Keep building - earlier context gets cached
conv.add(Message.user("What's the difference between append() and extend()?"))
resps3 = client.process_prompts_sync([conv], cache="last_user_message")
```

## Best Practices

1. **Cache Long, Stable Content**: Cache system prompts and tool definitions that don't change frequently.

2. **Choose the Right Pattern**: 
   - Use `system_and_tools` for consistent system prompts with tools
   - Use `tools_only` when system prompts vary but tools are constant
   - Use `last_user_message` for iterative refinement scenarios

3. **Monitor Usage**: Check `response.usage.has_cache_hit` and `response.usage.cache_read_tokens` to verify caching is working.

4. **Image Caching**: When caching conversations with images, the images are automatically locked as bytes to ensure cache consistency.

```python
# Images are automatically handled for caching
conv = Conversation.user("Analyze this chart").add_image("chart.png")

# Images get locked as bytes when cache is specified
resps = client.process_prompts_sync([conv], cache="last_user_message")
```

5. **Test Cache Patterns**: Different patterns work better for different use cases. Test to see which gives the best cost/performance trade-off for your specific scenario.

## Important Notes

- Prompt caching only works with Anthropic models (Claude)
- Other providers (OpenAI, etc.) will show a warning if you specify a cache parameter
- Bedrock Anthropic models also support prompt caching
- Cache hits are indicated in the usage statistics returned with each response
- Cached content must be identical to get cache hits - even small changes invalidate the cache