---
title: Installation
description: How to install and set up LM Deluge
---

## Installation

Install LM Deluge using pip:

```bash
pip install lm-deluge
```

## API Keys

The package relies on environment variables for API keys. We recommend using a `.env` file in your project root. `LLMClient` will automatically load the `.env` file when imported.

### Required Environment Variables

Depending on which providers you plan to use, set the appropriate API keys:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google AI Studio
GOOGLE_API_KEY=...

# Cohere
COHERE_API_KEY=...

# Meta
META_API_KEY=...

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### .env File

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Verification

Verify your installation by running a simple test:

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4o-mini")
resp = client.process_prompts_sync(["Say hello!"])
print(resp[0].completion)
```

If you see a response from the model, you're all set!

## Next Steps

Head over to the [Quick Start](/getting-started/quickstart/) guide to learn how to use LM Deluge effectively.
