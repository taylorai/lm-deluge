# Python examples of OpenAI-compatible + Anthropic-compatible API servers (and what to borrow)

This doc focuses on **server structure/scaffolding** (FastAPI routing, request validation, streaming, auth, error mapping, model routing) rather than inference internals. It includes a **drop-in skeleton** you can adapt to your own LLM SDK.

---

## Repos / projects worth skimming

### OpenAI-compatible servers (Python)

- **FastChat – OpenAI-compatible API server (FastAPI)**  
  Implements `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, streaming via **SSE** (`data: ...\n\n`, `[DONE]`). citeturn4view4turn4view3

- **vLLM – OpenAI-compatible server**  
  Has a dedicated OpenAI-compatible server entrypoint and documents auth middleware patterns. citeturn0search7turn0search3turn2search14

- **llama-cpp-python – OpenAI-compatible web server**  
  Provides a “drop-in replacement” OpenAI server (`python -m llama_cpp.server ...`). citeturn2search0turn16search12

- **LiteLLM Proxy – OpenAI-compatible gateway**  
  Proxy/gateway approach (OpenAI format in; many providers out). citeturn9search16turn0search10

- **Open WebUI – OpenAI-compatible endpoint(s)**  
  Exposes OpenAI-style chat completions endpoint (note: paths can differ, e.g. `/api/chat/completions`). citeturn2search2


### Anthropic-compatible servers/proxies (Python)

- **LiteLLM – `/v1/messages` Anthropic Messages API format**  
  Explicitly supports the Anthropic Messages request shape on `/v1/messages` (plus related endpoints). citeturn0search2turn0search35

- **claude-code-provider-proxy – Anthropic-compatible endpoints via FastAPI**  
  README describes exposing `POST /v1/messages` and `POST /v1/messages/count_tokens` with request/response translation to OpenAI-compatible providers. citeturn11view0turn1search8

- **claude-code-proxy – Anthropic client compatibility proxy (Python)**  
  Routes Anthropic clients (Claude Code) to OpenAI / Gemini / Anthropic backends, with model mapping rules. citeturn3view2turn1search3

- **mockllm (PyPI) – mock server for OpenAI + Anthropic formats**  
  Described as a FastAPI-based mock that mimics both formats (good for contract tests / scaffolding). citeturn1search16

- **mlx-omni-server – Anthropic-compatible endpoints**  
  Claims Anthropic-compatible endpoints including `/anthropic/v1/messages`. (May be helpful for shape + streaming patterns even if you won’t copy inference.) citeturn1search4


---

## Patterns to borrow (the useful scaffolding)

### 1) Route layer → adapter layer → provider/SDK layer
Most servers end up with a three-stage pipeline:

1. **FastAPI route**: validate request, auth, request-id, select streaming/non-streaming.  
2. **Adapter**: translate OpenAI/Anthropic JSON into your internal SDK request type.  
3. **Backend call**: call your SDK, return structured response; handle streaming token events.

FastChat is a clear example: the route validates, checks model, builds “gen_params”, then either returns a JSON response or a streaming generator (`StreamingResponse(..., media_type="text/event-stream")`). citeturn4view4turn4view3

### 2) Authentication as a FastAPI dependency / ASGI middleware
- FastChat uses a **dependency** (`Depends(check_api_key)`) on OpenAI-compatible endpoints. citeturn4view3turn3view1  
- vLLM documents an **ASGI middleware** that checks the Authorization Bearer token for `/v1/*` routes. citeturn0search3

### 3) Streaming = SSE everywhere (for compatibility)
OpenAI-compatible streaming is usually implemented as **Server-Sent Events**:

- Each chunk is sent as: `data: <json>\n\n`
- Final sentinel: `data: [DONE]\n\n`

FastChat does exactly this pattern. citeturn4view3

Anthropic streaming is also SSE, but with **different event types**; your server should emit the event objects expected by Anthropic clients.

### 4) “Model registry” endpoint and mapping
Compatibility clients often call a model listing endpoint early.
- In OpenAI-style: `/v1/models`
- Proxies often implement **model name remapping** (e.g., “sonnet” → “openai/gpt-4o”)

The claude-code-proxy README describes mapping Claude family names to configured OpenAI/Gemini models. citeturn3view2turn1search3

### 5) Error normalization
In practice, you want one internal exception type with fields like:
- `status_code` (HTTP)
- `error.type` / `error.code`
- `message`
Then map to:
- OpenAI error envelope (usually `{error: {message, type, ...}}`)
- Anthropic error envelope (different envelope + status codes)


---

## A practical “adapter server” skeleton (FastAPI + Pydantic)

This is **new code** (not copied from the repos above). It’s meant to be a clean starting point for:
- `POST /v1/chat/completions` (OpenAI Chat Completions)
- `POST /v1/messages` (Anthropic Messages)
- SSE streaming for both
- auth hooks for both

You’ll only need to implement 2 things:
- `YourLLMSDKClient` (your internal SDK wrapper)
- `openai_to_internal(...)` / `anthropic_to_internal(...)` (mapping logic)

### Suggested file layout

```text
your_adapter_server/
  app.py
  models_openai.py
  models_anthropic.py
  adapters.py
  sdk_client.py
```

---

## `models_openai.py` (minimal request/response models)

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, list[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class OpenAIToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIChatCompletionsRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False

    # common sampling knobs
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    # tool calling / function calling
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None

    # response formatting / json mode (optional)
    response_format: Optional[Dict[str, Any]] = None


class OpenAIChatCompletionsChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Optional[str] = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionsResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionsChoice]
    usage: Optional[OpenAIUsage] = None


# Streaming chunk shape (subset)
class OpenAIDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenAIChatCompletionsChunkChoice(BaseModel):
    index: int
    delta: OpenAIDelta
    finish_reason: Optional[str] = None


class OpenAIChatCompletionsChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatCompletionsChunkChoice]
```

---

## `models_anthropic.py` (minimal Messages API models)

```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel


class AnthropicContentBlock(BaseModel):
    type: Literal["text", "tool_use", "image"] = "text"
    text: Optional[str] = None

    # tool use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    stream: bool = False

    system: Optional[Union[str, list[AnthropicContentBlock]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # Anthropic-specific shape


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicContentBlock]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None
```

---

## `sdk_client.py` (your internal SDK wrapper interface)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class InternalTool:
    name: str
    description: Optional[str]
    json_schema: Optional[Dict[str, Any]]


@dataclass
class InternalMessage:
    role: str
    content: str


@dataclass
class InternalRequest:
    model: str
    messages: list[InternalMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[InternalTool]] = None


@dataclass
class InternalTokenEvent:
    text_delta: str = ""
    is_final: bool = False
    finish_reason: Optional[str] = None


@dataclass
class InternalResponse:
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class YourLLMSDKClient:
    async def generate(self, req: InternalRequest) -> InternalResponse:
        raise NotImplementedError

    async def stream(self, req: InternalRequest) -> AsyncIterator[InternalTokenEvent]:
        raise NotImplementedError
```

---

## `adapters.py` (request/response translation)

```python
from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from models_openai import (
    OpenAIChatCompletionsRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChatCompletionsChoice,
    OpenAIMessage,
    OpenAIUsage,
    OpenAIChatCompletionsChunk,
    OpenAIChatCompletionsChunkChoice,
    OpenAIDelta,
)
from models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicContentBlock,
    AnthropicUsage,
)
from sdk_client import InternalMessage, InternalRequest, InternalResponse, InternalTokenEvent


def _now_unix() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def openai_to_internal(req: OpenAIChatCompletionsRequest) -> InternalRequest:
    msgs = []
    for m in req.messages:
        if m.content is None:
            continue
        if isinstance(m.content, list):
            text = " ".join(str(part) for part in m.content)
        else:
            text = m.content
        msgs.append(InternalMessage(role=m.role, content=text))

    return InternalRequest(
        model=req.model,
        messages=msgs,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tools=None,
    )


def internal_to_openai_response(model: str, internal: InternalResponse) -> OpenAIChatCompletionsResponse:
    created = _now_unix()
    rid = _new_id("chatcmpl")

    msg = OpenAIMessage(role="assistant", content=internal.text)
    choice = OpenAIChatCompletionsChoice(index=0, message=msg, finish_reason=internal.finish_reason)
    usage = OpenAIUsage(
        prompt_tokens=internal.prompt_tokens,
        completion_tokens=internal.completion_tokens,
        total_tokens=internal.prompt_tokens + internal.completion_tokens,
    )
    return OpenAIChatCompletionsResponse(
        id=rid,
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )


async def internal_stream_to_openai_sse(
    model: str,
    events: AsyncIterator[InternalTokenEvent],
) -> AsyncIterator[str]:
    """
    Yields SSE lines for OpenAI streaming:
      data: {...}\n\n
      ...
      data: [DONE]\n\n
    """
    created = _now_unix()
    rid = _new_id("chatcmpl")

    first = OpenAIChatCompletionsChunk(
        id=rid,
        created=created,
        model=model,
        choices=[OpenAIChatCompletionsChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
    )
    yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"

    async for ev in events:
        chunk = OpenAIChatCompletionsChunk(
            id=rid,
            created=created,
            model=model,
            choices=[
                OpenAIChatCompletionsChunkChoice(
                    index=0,
                    delta=OpenAIDelta(content=ev.text_delta or None),
                    finish_reason=ev.finish_reason if ev.is_final else None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        if ev.is_final:
            break

    yield "data: [DONE]\n\n"


def anthropic_to_internal(req: AnthropicMessagesRequest) -> InternalRequest:
    msgs = []
    for m in req.messages:
        if isinstance(m.content, str):
            text = m.content
        else:
            text = "".join((b.text or "") for b in m.content if b.type == "text")
        msgs.append(InternalMessage(role=m.role, content=text))

    if req.system:
        if isinstance(req.system, str):
            sys_text = req.system
        else:
            sys_text = "".join((b.text or "") for b in req.system if b.type == "text")
        msgs.insert(0, InternalMessage(role="system", content=sys_text))

    return InternalRequest(
        model=req.model,
        messages=msgs,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tools=None,
    )


def internal_to_anthropic_response(model: str, internal: InternalResponse) -> AnthropicMessagesResponse:
    rid = _new_id("msg")
    content = [AnthropicContentBlock(type="text", text=internal.text)]
    usage = AnthropicUsage(input_tokens=internal.prompt_tokens, output_tokens=internal.completion_tokens)
    return AnthropicMessagesResponse(
        id=rid,
        model=model,
        content=content,
        stop_reason=internal.finish_reason,
        usage=usage,
    )


async def internal_stream_to_anthropic_sse(
    model: str,
    events: AsyncIterator[InternalTokenEvent],
) -> AsyncIterator[str]:
    """
    Anthropic streaming sends event objects. Clients usually accept SSE with:
      event: message_start
      data: {...}

    For broad compatibility: emit event+data lines.
    """
    rid = _new_id("msg")

    start = {
        "type": "message_start",
        "message": {"id": rid, "type": "message", "role": "assistant", "model": model, "content": []},
    }
    yield f"event: message_start\ndata: {start}\n\n"

    block_start = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    yield f"event: content_block_start\ndata: {block_start}\n\n"

    async for ev in events:
        if ev.text_delta:
            delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": ev.text_delta},
            }
            yield f"event: content_block_delta\ndata: {delta}\n\n"

        if ev.is_final:
            yield "event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n"
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": ev.finish_reason or "end_turn"},
                "usage": {"output_tokens": 0},
            }
            yield f"event: message_delta\ndata: {msg_delta}\n\n"
            yield "event: message_stop\ndata: {"type":"message_stop"}\n\n"
            break
```

---

## `app.py` (FastAPI routes for both compatibility modes)

```python
from __future__ import annotations

import os
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from models_openai import OpenAIChatCompletionsRequest
from models_anthropic import AnthropicMessagesRequest
from adapters import (
    openai_to_internal,
    internal_to_openai_response,
    internal_stream_to_openai_sse,
    anthropic_to_internal,
    internal_to_anthropic_response,
    internal_stream_to_anthropic_sse,
)
from sdk_client import YourLLMSDKClient


app = FastAPI(title="OpenAI+Anthropic Compatibility Adapter")


def require_openai_bearer(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("OPENAI_COMPAT_API_KEY")
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_anthropic_headers(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
) -> None:
    expected = os.getenv("ANTHROPIC_COMPAT_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid x-api-key")
    if os.getenv("REQUIRE_ANTHROPIC_VERSION") == "1" and not anthropic_version:
        raise HTTPException(status_code=400, detail="Missing anthropic-version header")


sdk = YourLLMSDKClient()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/v1/chat/completions", dependencies=[Depends(require_openai_bearer)])
async def openai_chat_completions(req: OpenAIChatCompletionsRequest):
    internal_req = openai_to_internal(req)

    if req.stream:
        events = sdk.stream(internal_req)
        return StreamingResponse(internal_stream_to_openai_sse(req.model, events), media_type="text/event-stream")

    resp = await sdk.generate(internal_req)
    return JSONResponse(content=internal_to_openai_response(req.model, resp).model_dump(exclude_none=True))


@app.post("/v1/messages", dependencies=[Depends(require_anthropic_headers)])
async def anthropic_messages(req: AnthropicMessagesRequest):
    internal_req = anthropic_to_internal(req)

    if req.stream:
        events = sdk.stream(internal_req)
        return StreamingResponse(internal_stream_to_anthropic_sse(req.model, events), media_type="text/event-stream")

    resp = await sdk.generate(internal_req)
    return JSONResponse(content=internal_to_anthropic_response(req.model, resp).model_dump(exclude_none=True))
```

---

## Quick contract tests

### OpenAI Chat Completions (non-streaming)

```bash
curl -s http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer test"   -d '{
    "model": "my-model",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

### OpenAI Chat Completions (streaming)

```bash
curl -N http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer test"   -d '{
    "model": "my-model",
    "stream": true,
    "messages": [{"role":"user","content":"Stream please"}]
  }'
```

### Anthropic Messages (non-streaming)

```bash
curl -s http://localhost:8000/v1/messages   -H "Content-Type: application/json"   -H "x-api-key: test"   -H "anthropic-version: 2023-06-01"   -d '{
    "model": "my-model",
    "max_tokens": 128,
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

### Anthropic Messages (streaming)

```bash
curl -N http://localhost:8000/v1/messages   -H "Content-Type: application/json"   -H "x-api-key: test"   -H "anthropic-version: 2023-06-01"   -d '{
    "model": "my-model",
    "max_tokens": 128,
    "stream": true,
    "messages": [{"role":"user","content":"Stream please"}]
  }'
```

---

## Notes / gotchas (from real-world servers)

- **SSE framing matters**: OpenAI clients/frameworks frequently require the `[DONE]` sentinel and JSON objects per `data:` line (FastChat demonstrates this). citeturn4view3
- **Auth routing**: vLLM’s middleware skips auth for non-`/v1` paths like `/health`, which is a nice pattern if you expose probes. citeturn0search3
- **Paths differ in the wild**: Open WebUI uses `/api/chat/completions` but keeps OpenAI-like body shape. If you’re trying to be “drop-in” for many clients, consider supporting aliases. citeturn2search2
- **Anthropic “count_tokens”**: many Claude/Agent tools call `/v1/messages/count_tokens` (provider proxies explicitly expose it). citeturn11view0turn1search8

---

## Where to go deeper

If you want to mirror a “production-grade” layout, skim these codebases specifically for:
- request validation models
- tool calling / tool call deltas
- streaming generators
- model listing responses
- error envelope shapes

FastChat is the easiest single-file read for OpenAI compatibility. citeturn4view4turn4view3  
LiteLLM is a good reference for “proxy as a product” (multi-provider, budgets, routing) and for `/v1/messages` coverage. citeturn0search35turn0search10
