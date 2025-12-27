---
title: Proxy Server
description: Deploy an OpenAI and Anthropic-compatible API proxy powered by lm-deluge
---

The LM-Deluge proxy server is a FastAPI-based reverse proxy that exposes OpenAI-compatible and Anthropic-compatible API endpoints. It allows you to route requests through lm-deluge's multi-provider support, apply model policies, and use a unified API regardless of which provider you're targeting.

## Quick Start

Install with the server extras:

```bash
pip install lm-deluge[server]
```

Start the server:

```bash
python -m lm_deluge.server
```

The server starts on `http://0.0.0.0:8000` by default.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models (OpenAI-compatible) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/messages` | POST | Anthropic-compatible messages |
| `/messages` | POST | Alternative Anthropic endpoint (SDK compatibility) |

## Using the Proxy

### With OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-proxy-key",  # Only if DELUGE_PROXY_API_KEY is set
)

response = client.chat.completions.create(
    model="claude-3.5-sonnet",  # Any model in the registry
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### With Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="your-proxy-key",  # Only if DELUGE_PROXY_API_KEY is set
)

response = client.messages.create(
    model="gpt-4o",  # Can use any model, even OpenAI models via Anthropic SDK
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
```

### With curl

```bash
# OpenAI format
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-key-here>" \
  -d '{
    "model": "claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Anthropic format
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-key-here>" \
  -d '{
    "model": "gpt-4o",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Command Line Options

```bash
python -m lm_deluge.server [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--host HOST` | Host to bind (default: `0.0.0.0`) |
| `--port PORT` | Port to run on (default: `8000`) |
| `--reload` | Enable auto-reload for development |
| `--config PATH` | Path to YAML config file |
| `--mode MODE` | Model policy mode: `allow_user_pick`, `force_default`, `alias_only` |
| `--allow-model MODEL` | Allow a specific model (can be repeated) |
| `--default-model MODEL` | Default model for `force_default` mode |
| `--routes JSON5` | JSON5 string defining route aliases |
| `--expose-aliases` | Show route aliases in `/v1/models` |
| `--hide-aliases` | Hide route aliases from `/v1/models` |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DELUGE_PROXY_HOST` | Host to bind (default: `0.0.0.0`) |
| `DELUGE_PROXY_PORT` | Port to run on (default: `8000`) |
| `DELUGE_PROXY_API_KEY` | API key clients must provide (optional) |
| `DELUGE_PROXY_TIMEOUT` | Request timeout in seconds (default: `120`) |
| `DELUGE_PROXY_LOG_REQUESTS` | Log incoming proxy requests |
| `DELUGE_PROXY_LOG_PROVIDER_REQUESTS` | Log outbound provider requests |
| `DELUGE_CACHE_PATTERN` | Cache pattern for Anthropic models |

### Cache Patterns

The `DELUGE_CACHE_PATTERN` environment variable controls prompt caching for Anthropic models:

- `tools_only` - Cache tools definition
- `system_and_tools` - Cache system prompt and tools
- `last_user_message` - Cache last user message
- `last_2_user_messages` - Cache last 2 user messages
- `last_3_user_messages` - Cache last 3 user messages

## Model Policies

Model policies control which models are exposed and how requests are routed.

### Policy Modes

**`allow_user_pick`** (default): Clients can request any allowed model.

```bash
python -m lm_deluge.server --mode allow_user_pick
```

**`force_default`**: All requests are routed to a default model regardless of what clients request.

```bash
python -m lm_deluge.server --mode force_default --default-model claude-3.5-sonnet
```

**`alias_only`**: Only exposes configured route aliases, hiding actual model names.

```bash
python -m lm_deluge.server --mode alias_only --routes '{"smart": {"models": ["claude-3.5-sonnet"]}}'
```

### Restricting Models

Limit which models can be used:

```bash
python -m lm_deluge.server \
  --allow-model claude-3.5-sonnet \
  --allow-model gpt-4o \
  --allow-model gpt-4o-mini
```

### Route Aliases

Route aliases let you expose friendly names that map to one or more backend models:

```bash
python -m lm_deluge.server --routes '{
  "fast": {"models": ["gpt-4o-mini", "claude-3.5-haiku"], "strategy": "round_robin"},
  "smart": {"models": ["claude-3.5-sonnet", "gpt-4o"], "strategy": "random"},
  "best": {"models": ["claude-3.5-sonnet"], "strategy": "round_robin"}
}'
```

Clients can then request `model: "fast"` and the proxy will route to one of the configured models.

### Route Strategies

- **`round_robin`**: Rotate through models in order
- **`random`**: Pick a random model each request
- **`weighted`**: Pick models based on weights

```bash
# Weighted routing: 70% to sonnet, 30% to gpt-4o
python -m lm_deluge.server --routes '{
  "smart": {
    "models": ["claude-3.5-sonnet", "gpt-4o"],
    "strategy": "weighted",
    "weights": [0.7, 0.3]
  }
}'
```

## YAML Configuration

For complex setups, use a YAML config file:

```yaml
# proxy-config.yaml
model_policy:
  mode: allow_user_pick
  allowed_models:
    - claude-3.5-sonnet
    - claude-3.5-haiku
    - gpt-4o
    - gpt-4o-mini
  expose_aliases: true
  routes:
    fast:
      models:
        - gpt-4o-mini
        - claude-3.5-haiku
      strategy: round_robin
    smart:
      models:
        - claude-3.5-sonnet
        - gpt-4o
      strategy: weighted
      weights:
        - 0.6
        - 0.4
```

Start with the config:

```bash
python -m lm_deluge.server --config proxy-config.yaml
```

## Authentication

By default, the proxy doesn't require authentication. To enable it, set the `DELUGE_PROXY_API_KEY` environment variable:

```bash
export DELUGE_PROXY_API_KEY="your-secret-key"
python -m lm_deluge.server
```

Clients must then provide the key:

- **OpenAI format**: `Authorization: Bearer your-secret-key`
- **Anthropic format**: `x-api-key: your-secret-key`

## Tool Support

The proxy automatically converts tools between OpenAI and Anthropic formats. You can send OpenAI-style tool definitions to the Anthropic endpoint or vice versa, and the proxy handles the conversion.

```python
# OpenAI SDK calling an Anthropic model with tools
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }]
)
```

## Limitations

- **No streaming**: The proxy currently does not support streaming responses. Set `stream=false` in your requests.
- **No embeddings**: Only chat/message completions are supported.

## Example: Development Setup

A typical development setup with logging and auto-reload:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DELUGE_PROXY_LOG_REQUESTS=1

python -m lm_deluge.server --reload --port 8080
```

## Example: Production Setup

A production setup with authentication and restricted models:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DELUGE_PROXY_API_KEY="my-proxy-secret"
export DELUGE_PROXY_TIMEOUT=300
export DELUGE_CACHE_PATTERN=system_and_tools

python -m lm_deluge.server \
  --host 0.0.0.0 \
  --port 8000 \
  --config /etc/deluge/proxy.yaml
```
