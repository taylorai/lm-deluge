import json
import os
import traceback as tb
from types import SimpleNamespace

import aiohttp
from aiohttp import ClientResponse

from lm_deluge.request_context import RequestContext
from lm_deluge.tool import MCPServer, Tool
from lm_deluge.warnings import maybe_warn
from lm_deluge.util.schema import (
    prepare_output_schema,
    transform_schema_for_openai,
)

from ..config import SamplingParams
from ..models import APIModel
from ..prompt import CachePattern, Conversation, Message, Text, Thinking, ToolCall
from ..usage import Usage
from .base import APIRequestBase, APIResponse


async def _build_oa_chat_request(
    model: APIModel,
    context: RequestContext,
) -> dict:
    prompt = context.prompt
    sampling_params = context.sampling_params
    tools = context.tools
    request_json = {
        "model": model.name,
        "messages": prompt.to_openai(),
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
    }
    if context.service_tier:
        assert context.service_tier in [
            "auto",
            "default",
            "flex",
            "priority",
        ], f"Invalid service tier: {context.service_tier}"
        # flex is only supported for o3, o4-mini, gpt-5 models
        if context.service_tier == "flex":
            model_supports_flex = any(x in model.id for x in ["o3", "o4-mini", "gpt-5"])
            if not model_supports_flex:
                print(
                    f"WARNING: service_tier='flex' only supported for o3, o4-mini, gpt-5. "
                    f"Using 'auto' instead for model {model.id}."
                )
                request_json["service_tier"] = "auto"
            else:
                request_json["service_tier"] = context.service_tier
        else:
            request_json["service_tier"] = context.service_tier
    # set max_tokens or max_completion_tokens dep. on provider
    if "cohere" in model.api_base:
        request_json["max_tokens"] = sampling_params.max_new_tokens
    else:
        request_json["max_completion_tokens"] = sampling_params.max_new_tokens
    if model.reasoning_model:
        request_json["temperature"] = 1.0
        request_json["top_p"] = 1.0
        effort = sampling_params.reasoning_effort
        if effort in [None, "none"]:
            # Disable reasoning for Gemini models when no effort requested
            if "gemini" in model.id:
                effort = "none"
            elif "gpt-5" in model.id:
                effort = "minimal"
            else:
                effort = "low"
        # GPT-5.1 models don't support 'minimal', they support 'none' instead
        if effort == "minimal" and "gpt-5.1" in model.id:
            maybe_warn("WARN_MINIMAL_TO_NONE", model_name=context.model_name)
            effort = "none"
        elif effort == "minimal" and "gpt-5" not in model.id:
            maybe_warn("WARN_MINIMAL_TO_LOW", model_name=context.model_name)
            effort = "low"
        request_json["reasoning_effort"] = effort
    else:
        if sampling_params.reasoning_effort:
            maybe_warn("WARN_REASONING_UNSUPPORTED", model_name=context.model_name)

    if sampling_params.logprobs:
        request_json["logprobs"] = True
        if sampling_params.top_logprobs is not None:
            request_json["top_logprobs"] = sampling_params.top_logprobs

    # Handle structured outputs (output_schema takes precedence over json_mode)
    if context.output_schema:
        if model.supports_json:
            base_schema = prepare_output_schema(context.output_schema)

            # Apply OpenAI-specific transformations (currently passthrough with copy)
            transformed_schema = transform_schema_for_openai(base_schema)

            request_json["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": transformed_schema,
                    "strict": True,
                },
            }
        else:
            print(
                f"WARNING: Model {model.name} does not support structured outputs. Ignoring output_schema."
            )
    elif sampling_params.json_mode and model.supports_json:
        request_json["response_format"] = {"type": "json_object"}

    if tools:
        request_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                request_tools.append(
                    tool.dump_for(
                        "openai-completions", strict=sampling_params.strict_tools
                    )
                )
            elif isinstance(tool, MCPServer):
                as_tools = await tool.to_tools()
                request_tools.extend(
                    [
                        t.dump_for(
                            "openai-completions", strict=sampling_params.strict_tools
                        )
                        for t in as_tools
                    ]
                )
        request_json["tools"] = request_tools
    return request_json


class OpenAIRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        # Pass context to parent, which will handle backwards compatibility
        super().__init__(context=context)

        # Warn if cache is specified for non-Anthropic model
        if self.context.cache is not None:
            maybe_warn(
                "WARN_CACHING_UNSUPPORTED",
                model_name=self.context.model_name,
                cache_param=self.context.cache,
            )
        self.model = APIModel.from_registry(self.context.model_name)

    async def build_request(self):
        self.url = f"{self.model.api_base}/chat/completions"
        base_headers = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic"]
        )

        self.request_json = await _build_oa_chat_request(self.model, self.context)

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        logprobs = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        finish_reason = None
        assert self.context.status_tracker

        if status_code == 500:
            print("Internal Server Error: ", (await http_response.text()))

        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()
            except Exception:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}"
                )
            if not is_error:
                assert data is not None, "data is None"
                try:
                    # Parse response into Message with parts
                    parts = []
                    message = data["choices"][0]["message"]
                    finish_reason = data["choices"][0]["finish_reason"]

                    # Add thinking content if present (reasoning models)
                    if "reasoning_content" in message:
                        thinking = message["reasoning_content"]
                        parts.append(Thinking(thinking))

                    # Together AI returns reasoning in a "reasoning"
                    # field which is not correct but whatever
                    if message.get("reasoning"):
                        thinking = message["reasoning"]
                        parts.append(Thinking(thinking))

                    # Add text content if present
                    if message.get("content"):
                        parts.append(Text(message["content"]))

                    # Add tool calls if present
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            parts.append(
                                ToolCall(
                                    id=tool_call["id"],
                                    name=tool_call["function"]["name"],
                                    arguments=json.loads(
                                        tool_call["function"]["arguments"]
                                    ),
                                )
                            )

                    content = Message("assistant", parts)

                    if "usage" in data and data["usage"] is not None:
                        usage = Usage.from_openai_usage(data["usage"])
                    if (
                        self.context.sampling_params.logprobs
                        and "logprobs" in data["choices"][0]
                    ):
                        logprobs = data["choices"][0]["logprobs"]["content"]
                except Exception:
                    is_error = True
                    error_message = f"Error getting 'choices' and 'usage' from {self.model.name} response."
        elif mimetype and "json" in mimetype.lower():
            is_error = True  # expected status is 200, otherwise it's an error
            data = await http_response.json()
            error_message = json.dumps(data)
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # handle special kinds of errors
        retry_with_different_model = status_code in [529, 429, 400, 401, 403, 413]
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower() or status_code == 429:
                error_message += " (Rate limit error, triggering cooldown.)"
                self.context.status_tracker.rate_limit_exceeded()
            if "context length" in error_message:
                error_message += " (Context length exceeded, set retries to 0.)"
                self.context.attempts_left = 0
            retry_with_different_model = True

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            logprobs=logprobs,
            thinking=thinking,
            content=content,
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=usage,
            raw_response=data,
            finish_reason=finish_reason,
            retry_with_different_model=retry_with_different_model,
        )


async def _build_oa_responses_request(
    model: APIModel,
    context: RequestContext,
):
    prompt = context.prompt
    sampling_params = context.sampling_params
    tools = context.tools
    openai_responses_format = prompt.to_openai_responses()
    request_json = {
        "model": model.name,
        "input": openai_responses_format["input"],
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "background": context.background or False,
    }
    if context.service_tier:
        assert context.service_tier in [
            "auto",
            "default",
            "flex",
            "priority",
        ], f"Invalid service tier: {context.service_tier}"
        # flex is only supported for o3, o4-mini, gpt-5 models
        if context.service_tier == "flex":
            model_supports_flex = any(x in model.id for x in ["o3", "o4-mini", "gpt-5"])
            if not model_supports_flex:
                print(
                    f"WARNING: service_tier='flex' only supported for o3, o4-mini, gpt-5. "
                    f"Model {model.id} doesn't support flex. Using 'auto' instead."
                )
                request_json["service_tier"] = "auto"
            else:
                request_json["service_tier"] = context.service_tier
        else:
            request_json["service_tier"] = context.service_tier
    if sampling_params.max_new_tokens:
        request_json["max_output_tokens"] = sampling_params.max_new_tokens

    if model.reasoning_model:
        effort = sampling_params.reasoning_effort
        if effort in [None, "none"]:
            # gemini models can switch reasoning off
            if "gemini" in model.id:
                effort = "none"
            else:
                effort = "low"
        # GPT-5.1 models don't support 'minimal', they support 'none' instead
        if effort == "minimal" and "gpt-5.1" in model.id:
            maybe_warn("WARN_MINIMAL_TO_NONE", model_name=context.model_name)
            effort = "none"
        elif effort == "minimal" and "gpt-5" not in model.id:
            maybe_warn("WARN_MINIMAL_TO_LOW", model_name=context.model_name)
            effort = "low"
        request_json["temperature"] = 1.0
        request_json["top_p"] = 1.0
        request_json["reasoning"] = {
            "effort": effort,
            "summary": "auto",
        }
    else:
        if sampling_params.reasoning_effort:
            maybe_warn("WARN_REASONING_UNSUPPORTED", model_name=context.model_name)

    # Handle structured outputs (output_schema takes precedence over json_mode)
    if context.output_schema:
        if model.supports_json:
            base_schema = prepare_output_schema(context.output_schema)

            # Apply OpenAI-specific transformations (currently passthrough with copy)
            transformed_schema = transform_schema_for_openai(base_schema)

            request_json["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": transformed_schema,
                    "strict": True,
                }
            }
        else:
            print(
                f"WARNING: Model {model.name} does not support structured outputs. Ignoring output_schema."
            )
    elif sampling_params.json_mode and model.supports_json:
        request_json["text"] = {"format": {"type": "json_object"}}

    # Handle tools
    request_tools = []
    # Add regular function tools
    for tool in tools or []:
        if isinstance(tool, Tool):
            request_tools.append(
                tool.dump_for("openai-responses", strict=sampling_params.strict_tools)
            )
        elif isinstance(tool, dict):
            # if computer use, make sure model supports it
            if tool["type"] == "computer_use_preview":
                if model.name != "openai-computer-use-preview":
                    raise ValueError(f"model {model.id} does not support computer use")
                # have to use truncation
                request_json["truncation"] = "auto"
            request_tools.append(tool)  # allow passing dict
        elif isinstance(tool, MCPServer):
            if context.force_local_mcp:
                as_tools = await tool.to_tools()
                request_tools.extend(
                    [
                        t.dump_for(
                            "openai-responses", strict=sampling_params.strict_tools
                        )
                        for t in as_tools
                    ]
                )
            else:
                request_tools.append(tool.for_openai_responses())

    if request_tools:
        request_json["tools"] = request_tools

    return request_json


class OpenAIResponsesRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context)
        # Warn if cache is specified for non-Anthropic model
        if self.context.cache is not None:
            maybe_warn(
                "WARN_CACHING_UNSUPPORTED",
                model_name=self.context.model_name,
                cache_param=self.context.cache,
            )
        self.model = APIModel.from_registry(self.context.model_name)

    async def build_request(self):
        self.url = f"{self.model.api_base}/responses"
        self.request_header = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }

        self.request_json = await _build_oa_responses_request(self.model, self.context)

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        logprobs = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        assert self.context.status_tracker

        if status_code == 500:
            res_text = await http_response.text()
            print("Internal Server Error: ", res_text)

        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()
            except Exception:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}"
                )
            if not is_error:
                assert data is not None, "data is None"

                # Check if response is incomplete
                if data.get("status") == "incomplete":
                    is_error = True
                    incomplete_reason = data.get("incomplete_details", {}).get(
                        "reason", "unknown"
                    )
                    error_message = f"Response incomplete: {incomplete_reason}"

                if not is_error:
                    try:
                        # Parse Responses API format
                        parts = []

                        # Get the output array from the response
                        output = data.get("output", [])
                        if not output:
                            is_error = True
                            error_message = f"No output in response. Status: {data.get('status')}, error: {data.get('error')}, incomplete details: {data.get('incomplete_details')}"
                        else:
                            # Process each output item
                            for item in output:
                                if item.get("type") == "message":
                                    message_content = item.get("content", [])
                                    for content_item in message_content:
                                        if content_item.get("type") == "output_text":
                                            parts.append(Text(content_item["text"]))
                                        elif content_item.get("type") == "refusal":
                                            parts.append(Text(content_item["refusal"]))
                                elif item.get("type") == "reasoning":
                                    summary = item["summary"]
                                    if not summary:
                                        continue
                                    if isinstance(summary, list) and len(summary) > 0:
                                        summary = summary[0]
                                    assert isinstance(
                                        summary, dict
                                    ), "summary isn't a dict"
                                    parts.append(Thinking(summary["text"]))
                                elif item.get("type") == "function_call":
                                    parts.append(
                                        ToolCall(
                                            id=item["call_id"],
                                            name=item["name"],
                                            arguments=json.loads(item["arguments"]),
                                        )
                                    )
                                elif item.get("type") == "mcp_call":
                                    parts.append(
                                        ToolCall(
                                            id=item["id"],
                                            name=item["name"],
                                            arguments=json.loads(item["arguments"]),
                                            built_in=True,
                                            built_in_type="mcp_call",
                                            extra_body={
                                                "server_label": item["server_label"],
                                                "error": item.get("error"),
                                                "output": item.get("output"),
                                            },
                                        )
                                    )

                                elif item.get("type") == "computer_call":
                                    parts.append(
                                        ToolCall(
                                            id=item["call_id"],
                                            name="computer_call",
                                            arguments=item.get("action"),
                                            built_in=True,
                                            built_in_type="computer_call",
                                        )
                                    )

                                elif item.get("type") == "web_search_call":
                                    parts.append(
                                        ToolCall(
                                            id=item["id"],
                                            name="web_search_call",
                                            arguments={},
                                            built_in=True,
                                            built_in_type="web_search_call",
                                            extra_body={"status": item["status"]},
                                        )
                                    )

                                elif item.get("type") == "file_search_call":
                                    parts.append(
                                        ToolCall(
                                            id=item["id"],
                                            name="file_search_call",
                                            arguments={"queries": item["queries"]},
                                            built_in=True,
                                            built_in_type="file_search_call",
                                            extra_body={
                                                "status": item["status"],
                                                "results": item["results"],
                                            },
                                        )
                                    )
                                elif item.get("type") == "image_generation_call":
                                    parts.append(
                                        ToolCall(
                                            id=item["id"],
                                            name="image_generation_call",
                                            arguments={},
                                            built_in=True,
                                            built_in_type="image_generation_call",
                                            extra_body={
                                                "status": item["status"],
                                                "result": item["result"],
                                            },
                                        )
                                    )

                            # Handle reasoning if present
                            if "reasoning" in data and data["reasoning"].get("summary"):
                                thinking = data["reasoning"]["summary"]
                                parts.append(Thinking(thinking))

                            content = Message("assistant", parts)

                            # Extract usage information
                            if "usage" in data and data["usage"] is not None:
                                usage = Usage.from_openai_usage(data["usage"])

                    except Exception as e:
                        is_error = True
                        error_message = f"Error parsing {self.model.name} responses API response: {str(e)}"
                        print("got data:", data)
                        traceback = tb.format_exc()
                        print(f"Error details:\n{traceback}")

        elif mimetype and "json" in mimetype.lower():
            print("is_error True, json response")
            is_error = True
            data = await http_response.json()
            error_message = json.dumps(data)
        else:
            print("is_error True, non-json response")
            is_error = True
            text = await http_response.text()
            error_message = text

        # Handle special kinds of errors
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower() or status_code == 429:
                error_message += " (Rate limit error, triggering cooldown.)"
                self.context.status_tracker.rate_limit_exceeded()
            if "context length" in error_message:
                error_message += " (Context length exceeded, set retries to 0.)"
                self.context.attempts_left = 0

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            logprobs=logprobs,
            thinking=thinking,
            content=content,
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=usage,
            raw_response=data,
        )


async def stream_chat(
    model_name: str,  # must correspond to registry
    prompt: Conversation,
    sampling_params: SamplingParams = SamplingParams(),
    tools: list | None = None,
    cache: CachePattern | None = None,
    extra_headers: dict[str, str] | None = None,
):
    if cache is not None:
        maybe_warn(
            "WARN_CACHING_UNSUPPORTED",
            model_name=model_name,
            cache_param=cache,
        )

    model = APIModel.from_registry(model_name)
    if model.api_spec != "openai":
        raise ValueError("streaming only supported on openai models for now")
    url = f"{model.api_base}/chat/completions"
    base_headers = {"Authorization": f"Bearer {os.getenv(model.api_key_env_var)}"}

    # Merge extra headers, filtering out anthropic headers
    request_header = dict(base_headers)
    if extra_headers:
        filtered_extra = {
            k: v for k, v in extra_headers.items() if "anthropic" not in k.lower()
        }
        request_header.update(filtered_extra)

    context = SimpleNamespace(
        prompt=prompt, tools=tools, sampling_params=sampling_params
    )

    request_json = await _build_oa_chat_request(model, context)  # type: ignore
    request_json["stream"] = True

    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=request_header, json=request_json) as r:
            r.raise_for_status()  # bail on 4xx/5xx
            content = ""
            buf = ""
            async for chunk in r.content.iter_any():  # raw bytes
                buf += chunk.decode()
                while "\n\n" in buf:  # full SSE frame
                    event, buf = buf.split("\n\n", 1)
                    if not event.startswith("data:"):
                        continue  # ignore comments
                    data = event[5:].strip()  # after "data:"
                    if data == "[DONE]":
                        yield APIResponse(
                            id=0,
                            status_code=None,
                            is_error=False,
                            error_message=None,
                            prompt=prompt,
                            content=Message(
                                role="assistant", parts=[Text(text=content)]
                            ),
                            model_internal=model.id,
                            sampling_params=sampling_params,
                            usage=None,
                            raw_response=None,
                        )
                    msg = json.loads(data)  # SSE payload
                    delta = msg["choices"][0]["delta"].get("content")
                    if delta:
                        content += delta
                        yield delta
