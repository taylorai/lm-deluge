---
title: Installation
description: How to install and set up LM Deluge
---

## Installation

Install LM Deluge using pip (Python 3.10+):

```bash
python -m pip install -U lm-deluge
```

Optional extras:

- `pip install plyvel` if you want LevelDB-backed local caching
- `pip install pdf2image pillow` if you plan to turn PDFs into images via `Image.from_pdf`

## API Keys

LM Deluge reads API keys from environment variables so the client can contact each provider directly. Load them at process startup (for example with [`python-dotenv`](https://pypi.org/project/python-dotenv/)) and pass the values down to your workers, CLI scripts, or notebook kernels.

### Required Environment Variables

Depending on which providers you plan to use, set the appropriate API keys:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google AI Studio (Gemini)
GEMINI_API_KEY=...

# Cohere
COHERE_API_KEY=...

# OpenRouter (any model prefixed with openrouter:)
OPENROUTER_API_KEY=...

# Meta, Groq, DeepSeek, etc. use provider-specific keys defined in src/lm_deluge/models

# AWS Bedrock (for Amazon-hosted Anthropic and Meta models)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### .env File

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

Load the file inside your application:

```python
import dotenv

dotenv.load_dotenv()
```

## Verification

Verify your installation by running a simple test:

```python
import dotenv
from lm_deluge import LLMClient

dotenv.load_dotenv()

client = LLMClient("gpt-4.1-mini")
responses = client.process_prompts_sync(["Say hello!"])
print(responses[0].completion)
```

If you see a response from the model, you're all set!

## Skill for AI Coding Assistants

If you use AI coding assistants like Claude Code or Codex, you can install the lm-deluge skill to give them instant familiarity with the library:

```bash
# Install to default location (~/.claude/skills)
deluge skill install

# Or specify a custom directory
deluge skill install ~/.codex/skills
```

This installs a `SKILL.md` file that teaches your coding assistant how to use lm-deluge correctly—including the right imports, common patterns, and API gotchas—so it can help you write code that works on the first try.

## Next Steps

Head over to the [Quick Start](/getting-started/quickstart/) guide to learn how to use LM Deluge effectively.
