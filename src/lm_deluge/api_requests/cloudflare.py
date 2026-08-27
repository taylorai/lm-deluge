import os
from typing import Any

from .moondream import (
    MoondreamRequest,
    _first_image,
    _image_url,
    _last_user_text,
    _query_image_and_text,
)
from .openai import OpenAIRequest, _build_oa_chat_request


class CloudflareRequest(OpenAIRequest):
    """OpenAI-compatible handler for Cloudflare Workers AI.

    The only difference from vanilla OpenAI is that the api_base URL contains
    a ``{account_id}`` placeholder which must be resolved from the
    ``CLOUDFLARE_ACCOUNT_ID`` environment variable at request time, and
    Cloudflare uses ``max_tokens`` instead of ``max_completion_tokens``.
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

        # Cloudflare doesn't support reasoning_effort
        self.request_json.pop("reasoning_effort", None)


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
