import json
import os

from aiohttp import ClientResponse

from lm_deluge.request_context import RequestContext
from lm_deluge.tool import Tool
from lm_deluge.warnings import maybe_warn

from ..config import SamplingParams
from ..models import APIModel
from ..prompt import Conversation, Message, Text, ThoughtSignature, Thinking, ToolCall
from ..usage import Usage
from .base import APIRequestBase, APIResponse


async def _build_gemini_request(
    model: APIModel,
    prompt: Conversation,
    tools: list[Tool] | None,
    sampling_params: SamplingParams,
) -> dict:
    system_message, messages = prompt.to_gemini()

    # For Gemini 3, inject dummy signatures when missing for function calls
    is_gemini_3 = "gemini-3" in model.name.lower()
    if is_gemini_3:
        dummy_sig = "context_engineering_is_the_way_to_go"
        for msg in messages:
            if "parts" in msg:
                for part in msg["parts"]:
                    # For function calls, inject dummy signature if missing
                    if "functionCall" in part and "thoughtSignature" not in part:
                        part["thoughtSignature"] = dummy_sig
                        maybe_warn(
                            "WARN_GEMINI3_MISSING_SIGNATURE",
                            part_type="function call",
                        )

    request_json = {
        "contents": messages,
        "generationConfig": {
            "temperature": sampling_params.temperature,
            "topP": sampling_params.top_p,
            "maxOutputTokens": sampling_params.max_new_tokens,
        },
    }

    # Add system instruction if present
    if system_message:
        request_json["systemInstruction"] = {"parts": [{"text": system_message}]}

    # Handle reasoning models (thinking)
    is_gemini_3 = "gemini-3" in model.name.lower()
    is_gemini_3_flash = "gemini-3-flash" in model.name.lower()
    if is_gemini_3:
        # gemini3 MUST think
        if not sampling_params.reasoning_effort:
            maybe_warn("WARN_GEMINI3_NO_REASONING")
            effort = "low"
        else:
            effort_key = sampling_params.reasoning_effort
            if effort_key == "xhigh":
                maybe_warn("WARN_XHIGH_TO_HIGH", model_name=model.name)
                effort_key = "high"
            if is_gemini_3_flash:
                # Flash supports minimal, low, medium, high
                level_map = {
                    "none": "low",
                    "minimal": "minimal",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                }
            else:
                # Pro only supports low, high
                level_map = {
                    "none": "low",
                    "minimal": "low",
                    "low": "low",
                    "medium": "high",
                    "high": "high",
                }
            effort = level_map[effort_key]
        thinking_config = {"thinkingLevel": effort}
        request_json["generationConfig"]["thinkingConfig"] = thinking_config

    elif model.reasoning_model:
        if (
            sampling_params.thinking_budget is not None
            and sampling_params.reasoning_effort is not None
        ):
            maybe_warn("WARN_THINKING_BUDGET_AND_REASONING_EFFORT")

        if (
            sampling_params.thinking_budget is not None
            and sampling_params.thinking_budget > 0
        ):
            thinking_config = {
                "includeThoughts": True,
                "thinkingBudget": sampling_params.thinking_budget,
            }
        elif sampling_params.thinking_budget == -1:
            # dynamic thinking
            thinking_config = {"includeThoughts": True, "thinkingBudget": -1}
        elif sampling_params.reasoning_effort not in [None, "none"]:
            effort_key = sampling_params.reasoning_effort
            if effort_key == "xhigh":
                maybe_warn("WARN_XHIGH_TO_HIGH", model_name=model.name)
                effort_key = "high"
            level_map = {
                "minimal": 256,
                "low": 1024,
                "medium": 4096,
                "high": 16384,
            }
            assert effort_key in level_map
            budget = level_map[effort_key]
            if "flash-lite" in model.id:
                budget = max(budget, 512)
            thinking_config = {"includeThoughts": True, "thinkingBudget": budget}
        elif "2.5-pro" in model.id:
            # 2.5 pro must think.
            thinking_config = {"includeThoughts": True, "thinkingBudget": 128}
        else:
            # no thoughts head empty
            thinking_config = {"includeThoughts": False, "thinkingBudget": 0}

        request_json["generationConfig"]["thinkingConfig"] = thinking_config

    else:
        if sampling_params.reasoning_effort:
            maybe_warn("WARN_REASONING_UNSUPPORTED", model_name=model.name)

    # Add tools if provided
    if tools:
        request_tools = []
        function_declarations = []

        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "gemini_computer_use":
                # Gemini computer use tool - add as separate tool entry
                env_map = {
                    "browser": "ENVIRONMENT_BROWSER",
                    "android": "ENVIRONMENT_ANDROID",
                }
                env = env_map.get(
                    tool.get("environment", "browser"), "ENVIRONMENT_BROWSER"
                )
                cu_tool: dict = {
                    "computerUse": {
                        "environment": env,
                    }
                }
                excluded = tool.get("excluded_predefined_functions")
                if excluded:
                    cu_tool["computerUse"]["excludedPredefinedFunctions"] = excluded
                request_tools.append(cu_tool)
            elif hasattr(tool, "dump_for"):
                # Regular Tool object
                function_declarations.append(tool.dump_for("google"))
            elif isinstance(tool, dict):
                # Raw dict tool - assume it's a function declaration
                function_declarations.append(tool)

        if function_declarations:
            request_tools.append({"functionDeclarations": function_declarations})

        if request_tools:
            request_json["tools"] = request_tools

    # Handle JSON mode
    if sampling_params.json_mode and model.supports_json:
        request_json["generationConfig"]["responseMimeType"] = "application/json"

    # Handle media_resolution for Gemini 3 (requires v1alpha)
    if sampling_params.media_resolution is not None:
        is_gemini_3 = "gemini-3" in model.name.lower()
        if is_gemini_3:
            # Add global media resolution to generationConfig
            request_json["generationConfig"]["mediaResolution"] = {
                "level": sampling_params.media_resolution
            }
        else:
            # Warn if trying to use media_resolution on non-Gemini-3 models
            maybe_warn(
                "WARN_MEDIA_RESOLUTION_UNSUPPORTED",
                model_name=model.name,
            )

    return request_json


class GeminiRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        # Warn if cache is specified for Gemini model
        if self.context.cache is not None:
            maybe_warn(
                "WARN_CACHING_UNSUPPORTED",
                model_name=self.context.model_name,
                cache_param=self.context.cache,
            )

        self.model = APIModel.from_registry(self.context.model_name)

    async def build_request(self):
        self.url = f"{self.model.api_base}/models/{self.model.name}:generateContent"
        base_headers = {
            "Content-Type": "application/json",
        }
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic", "openai", "mistral"]
        )

        # Add API key as query parameter for Gemini
        api_key = os.getenv(self.model.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key environment variable {self.model.api_key_env_var} not set"
            )
        self.url += f"?key={api_key}"

        self.request_json = await _build_gemini_request(
            self.model,
            self.context.prompt,
            self.context.tools,  # type: ignore
            self.context.sampling_params,
        )

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        assert self.context.status_tracker

        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()
            except Exception as e:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}: {e}"
                )

            if not is_error:
                assert data
                try:
                    # Parse Gemini response format
                    parts = []

                    if "candidates" in data and data["candidates"]:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                # Extract thought signature if present
                                raw_sig = part.get("thoughtSignature")
                                thought_sig = (
                                    ThoughtSignature(raw_sig, provider="gemini")
                                    if raw_sig is not None
                                    else None
                                )

                                if "text" in part:
                                    parts.append(
                                        Text(
                                            part["text"],
                                            thought_signature=thought_sig,
                                        )
                                    )
                                elif "thought" in part:
                                    # Thought with optional signature
                                    parts.append(
                                        Thinking(
                                            content=part["thought"],
                                            thought_signature=thought_sig,
                                        )
                                    )
                                elif "functionCall" in part:
                                    func_call = part["functionCall"]
                                    # Generate a unique ID since Gemini doesn't provide one
                                    import uuid

                                    tool_id = f"call_{uuid.uuid4().hex[:8]}"
                                    parts.append(
                                        ToolCall(
                                            id=tool_id,
                                            name=func_call["name"],
                                            arguments=func_call.get("args", {}),
                                            thought_signature=thought_sig,
                                        )
                                    )
                                elif thought_sig:
                                    parts.append(
                                        Text("", thought_signature=thought_sig)
                                    )

                    content = Message("assistant", parts)

                    # Extract usage information if present
                    if "usageMetadata" in data:
                        usage_data = data["usageMetadata"]
                        usage = Usage.from_gemini_usage(usage_data)

                except Exception as e:
                    is_error = True
                    error_message = f"Error parsing Gemini response: {str(e)}"

        elif mimetype and "json" in mimetype.lower():
            is_error = True
            try:
                data = await http_response.json()
                error_message = json.dumps(data)
            except Exception:
                error_message = (
                    f"HTTP {status_code} with JSON content type but failed to parse"
                )
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # Handle special kinds of errors
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower() or status_code == 429:
                error_message += " (Rate limit error, triggering cooldown.)"
                self.context.status_tracker.rate_limit_exceeded()
            if (
                "context length" in error_message.lower()
                or "token limit" in error_message.lower()
            ):
                error_message += " (Context length exceeded, set retries to 0.)"
                self.context.attempts_left = 0

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            content=content,
            thinking=thinking,
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=usage,
            raw_response=data,
        )
