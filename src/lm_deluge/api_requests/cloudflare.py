import json
import os
from typing import Any

from aiohttp import ClientResponse

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Image, Message, Text
from lm_deluge.usage import Usage

from .base import APIRequestBase, APIResponse
from .moondream import (
    MoondreamRequest,
    _first_image,
    _image_url,
    _last_user_text,
    _query_image_and_text,
)
from .openai import OpenAIRequest, _build_oa_chat_request

# Models whose top reasoning level isn't called "high" (Qwen rejects "high").
_HIGH_EFFORT_ALIASES = {"@cf/qwen/qwen3.8-27b": "xhigh"}


class CloudflareRequest(OpenAIRequest):
    """OpenAI-compatible handler for Cloudflare Workers AI.

    The only difference from vanilla OpenAI is that the api_base URL contains
    a ``{account_id}`` placeholder which must be resolved from the
    ``CLOUDFLARE_ACCOUNT_ID`` environment variable at request time,
    Cloudflare uses ``max_tokens`` instead of ``max_completion_tokens``, and
    ``reasoning_effort`` is only sent when explicitly requested.
    """

    async def build_request(self):
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required "
                "for Cloudflare Workers AI models."
            )

        base = self.model.api_base.replace("{account_id}", account_id)
        self.url = f"{base}/chat/completions"

        base_headers = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic"]
        )

        self.request_json = await _build_oa_chat_request(self.model, self.context)

        # Cloudflare uses max_tokens, not max_completion_tokens
        if "max_completion_tokens" in self.request_json:
            self.request_json["max_tokens"] = self.request_json.pop(
                "max_completion_tokens"
            )

        # Only send reasoning_effort when the caller set one, so models keep
        # their own defaults. Unsupported levels were already mapped down via
        # the model's supports_* flags.
        if self.context.sampling_params.reasoning_effort is None:
            self.request_json.pop("reasoning_effort", None)
        elif self.request_json.get("reasoning_effort") == "high":
            self.request_json["reasoning_effort"] = _HIGH_EFFORT_ALIASES.get(
                self.model.name, "high"
            )


class CloudflareMoondreamRequest(MoondreamRequest):
    """Cloudflare-hosted Moondream adapter for its model-specific schema."""

    async def build_request(self):
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required "
                "for Cloudflare Workers AI models."
            )

        extra = dict(self.context.extra_body or {})
        self.task = extra.pop("task", "query")
        if self.task not in {"caption", "detect", "point", "query"}:
            raise ValueError(
                "Unsupported Cloudflare Moondream task: "
                f"{self.task}. Expected one of ['caption', 'detect', 'point', 'query']."
            )
        if extra.pop("stream", False):
            raise NotImplementedError(
                "Cloudflare Moondream streaming is not supported yet."
            )

        prompt = self.context.prompt
        image = _first_image(prompt)
        text = _last_user_text(prompt)
        if self.task == "query":
            image, text = _query_image_and_text(prompt)
        elif image is None:
            raise ValueError(f"Moondream task '{self.task}' requires an image.")

        payload = {
            "task": self.task,
            "image": _image_url(image),
            "stream": False,
            "max_tokens": self.context.sampling_params.max_new_tokens,
        }
        sampling = self.context.sampling_params
        if sampling.temperature != 1.0:
            payload["temperature"] = sampling.temperature
        if sampling.top_p != 1.0:
            payload["top_p"] = sampling.top_p

        if self.task == "query":
            question = extra.pop("question", text)
            if not question:
                raise ValueError("Moondream query requires a question or user text.")
            payload["question"] = question
            reasoning = extra.pop("reasoning", None)
            if reasoning is None:
                reasoning = sampling.reasoning_effort not in {None, "none"}
            payload["reasoning"] = reasoning
        elif self.task == "caption":
            payload["caption_length"] = extra.pop("caption_length", "normal")
        else:
            target = extra.pop("target", None)
            if not target:
                raise ValueError(
                    f"Moondream task '{self.task}' requires extra_body['target']."
                )
            payload["target"] = target

        max_objects = extra.pop("max_objects", None)
        if max_objects is not None:
            payload["max_objects"] = max_objects
        payload.update(extra)

        base = self.model.api_base.replace("{account_id}", account_id)
        self.url = f"{base}/{self.model.name}"
        self.request_header = self.merge_headers(
            {
                "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}",
                "Content-Type": "application/json",
            }
        )
        self.request_json = payload

    def unwrap_response(self, data: dict[str, Any]) -> dict[str, Any]:
        while isinstance(data.get("result"), dict):
            data = data["result"]

        for key in ("output", "response"):
            nested = data.get(key)
            if isinstance(nested, dict):
                data = nested
                break

        text = data.get("response")
        if not isinstance(text, str):
            text = data.get("output")
        if isinstance(text, str):
            completion_key = "caption" if self.task == "caption" else "answer"
            if not data.get(completion_key):
                data = {**data, completion_key: text}

        return data


def _clef_state(prompt: Conversation) -> str | list[dict[str, str]]:
    """Use a lone message's text as the state, else a role/content transcript."""
    turns = []
    for message in prompt.messages:
        text = "\n".join(
            part.text for part in message.parts if isinstance(part, Text)
        ).strip()
        if text:
            turns.append({"role": message.role, "content": text})
    if not turns:
        raise ValueError(
            "Clef requires text in the conversation or extra_body['state']."
        )
    if len(turns) == 1:
        return turns[0]["content"]
    return turns


class CloudflareClefRequest(APIRequestBase):
    """Adapter for Cloudflare's Clef decision models (System One API).

    Questions must be passed as ``extra_body["questions"]``. The state is the
    conversation text unless ``extra_body["state"]`` is given, and Image parts
    are sent as embedded data URLs. Other ``extra_body`` keys (e.g.
    ``audio`` for clef-omni) are passed through. The completion is the JSON
    ``answers`` object; the full result is in ``raw_response``.
    """

    def __init__(self, context: RequestContext):
        super().__init__(context=context)
        self.model = APIModel.from_registry(self.context.model_name)

    async def build_request(self):
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required "
                "for Cloudflare Workers AI models."
            )

        extra = dict(self.context.extra_body or {})
        if extra.pop("stream", False):
            raise NotImplementedError("Clef does not support streaming.")
        questions = extra.pop("questions", None)
        if not questions:
            raise ValueError("Clef requires extra_body['questions'].")

        prompt = self.context.prompt
        state = extra.pop("state", None)
        if state is None:
            state = _clef_state(prompt)

        payload: dict[str, Any] = {
            "model": self.model.name.rsplit("/", 1)[-1],
            "state": state,
            "questions": questions,
        }
        images = list(extra.pop("images", []))
        for message in prompt.messages:
            for part in message.parts:
                if isinstance(part, Image):
                    images.append(part._base64())
        if images:
            payload["images"] = images
        payload.update(extra)

        base = self.model.api_base.replace("{account_id}", account_id)
        self.url = f"{base}/{self.model.name}"
        self.request_header = self.merge_headers(
            {
                "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}",
                "Content-Type": "application/json",
            }
        )
        self.request_json = payload

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", "")
        data = None
        is_error = False
        error_message = None
        content = None
        usage = None

        if 200 <= status_code < 300:
            try:
                data = await http_response.json()
            except Exception:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}"
                )
            if not is_error:
                assert data is not None
                if data.get("success") is False:
                    is_error = True
                    error_message = json.dumps(data.get("errors") or data)
                else:
                    data = data.get("result", data)
                    answers = data.get("answers")
                    if not isinstance(answers, dict):
                        is_error = True
                        error_message = f"Clef response has no answers: {data!r}"
                    else:
                        content = Message("assistant", [Text(json.dumps(answers))])
                        raw_usage = data.get("usage") or {}
                        usage = Usage(
                            input_tokens=raw_usage.get("input_tokens", 0),
                            output_tokens=raw_usage.get("output_tokens", 0),
                        )
        elif "json" in mimetype.lower():
            is_error = True
            text = await http_response.text()
            try:
                data = json.loads(text)
                error_message = json.dumps(data)
            except Exception:
                error_message = text
        else:
            is_error = True
            error_message = await http_response.text()

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            content=content,
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=usage,
            raw_response=data,
            retry_with_different_model=status_code in [429, 500, 502, 503, 504],
            give_up_if_no_other_models=status_code in [401, 403, 404],
        )
