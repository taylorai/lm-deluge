import json
import os
from typing import Any

from aiohttp import ClientResponse

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Image, Message, Text

from .base import APIRequestBase, APIResponse


MOONDREAM_TASKS = {"caption", "query", "detect", "point", "segment"}


def _first_image(prompt: Conversation) -> Image | None:
    for message in prompt.messages:
        for part in message.parts:
            if isinstance(part, Image):
                return part
    return None


def _query_image_and_text(prompt: Conversation) -> tuple[Image, str]:
    images: list[Image] = []
    texts: list[str] = []

    for message in prompt.messages:
        if message.role != "user":
            continue
        for part in message.parts:
            if isinstance(part, Image):
                images.append(part)
            elif isinstance(part, Text) and part.text.strip():
                texts.append(part.text.strip())

    if len(images) != 1 or len(texts) != 1:
        raise ValueError(
            "Moondream query requires exactly one user image part and exactly "
            f"one non-empty user text part; got {len(images)} image parts and "
            f"{len(texts)} text parts."
        )
    return images[0], texts[0]


def _last_user_text(prompt: Conversation) -> str:
    for message in reversed(prompt.messages):
        if message.role != "user":
            continue
        texts = [part.text for part in message.parts if isinstance(part, Text)]
        if texts:
            return "\n".join(texts).strip()
    return ""


def _image_url(image: Image) -> str:
    payload = image.oa_chat()["image_url"]
    return payload["url"]


def _settings_from_context(context: RequestContext) -> dict[str, Any]:
    sampling = context.sampling_params
    settings: dict[str, Any] = {"max_tokens": sampling.max_new_tokens}
    if sampling.temperature != 1.0:
        settings["temperature"] = sampling.temperature
    if sampling.top_p != 1.0:
        settings["top_p"] = sampling.top_p
    return settings


async def _build_moondream_request(
    model: APIModel,
    context: RequestContext,
) -> tuple[str, dict[str, Any]]:
    extra = dict(context.extra_body or {})
    task = extra.pop("task", "query")
    if task not in MOONDREAM_TASKS:
        raise ValueError(
            f"Unsupported Moondream task: {task}. Expected one of {sorted(MOONDREAM_TASKS)}."
        )
    if extra.pop("stream", False):
        raise NotImplementedError("Moondream streaming is not supported yet.")

    prompt = context.prompt
    image = _first_image(prompt)
    text = _last_user_text(prompt)

    payload: dict[str, Any] = {}
    if task == "query":
        image, text = _query_image_and_text(prompt)
        payload["image_url"] = _image_url(image)
    else:
        if image is None:
            raise ValueError(f"Moondream task '{task}' requires an image.")
        payload["image_url"] = _image_url(image)

    if task == "caption":
        payload["length"] = extra.pop("length", "normal")
    elif task == "query":
        question = extra.pop("question", text)
        if not question:
            raise ValueError("Moondream query requires a question or user text.")
        payload["question"] = question
        reasoning = extra.pop("reasoning", None)
        if reasoning is None:
            reasoning = context.sampling_params.reasoning_effort not in {None, "none"}
        if reasoning:
            payload["reasoning"] = True
    elif task in {"detect", "point", "segment"}:
        obj = extra.pop("object", None)
        if not obj:
            raise ValueError(f"Moondream task '{task}' requires extra_body['object'].")
        payload["object"] = obj
        if task == "segment":
            payload["stream"] = False
            spatial_refs = extra.pop("spatial_refs", None)
            if spatial_refs is not None:
                payload["spatial_refs"] = spatial_refs

    max_objects = extra.pop("max_objects", None)
    settings = extra.pop("settings", None)
    if settings is None:
        settings = _settings_from_context(context)
    if max_objects is not None:
        settings["max_objects"] = max_objects
    if settings:
        payload["settings"] = settings

    api_model = extra.pop("model", None)
    if api_model:
        payload["model"] = api_model

    payload.update(extra)
    return task, payload


class MoondreamRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)
        self.model = APIModel.from_registry(self.context.model_name)
        self.task = "query"

    async def build_request(self):
        self.task, self.request_json = await _build_moondream_request(
            self.model, self.context
        )
        self.url = f"{self.model.api_base}/{self.task}"
        base_headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        api_key = os.getenv(self.model.api_key_env_var)
        if api_key is not None:
            base_headers["X-Moondream-Auth"] = api_key
        self.request_header = self.merge_headers(base_headers)

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", "")
        data = None
        is_error = False
        error_message = None
        content = None

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
                text = None
                if self.task == "caption":
                    text = data.get("caption")
                elif self.task == "query":
                    text = data.get("answer")
                else:
                    text = json.dumps(data)
                if text is not None:
                    content = Message("assistant", [Text(text)])
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

        retry_with_different_model = status_code in [429, 500, 502, 503, 504]
        give_up_if_no_other_models = status_code in [401, 403, 404]

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            content=content,
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=None,
            raw_response=data,
            retry_with_different_model=retry_with_different_model,
            give_up_if_no_other_models=give_up_if_no_other_models,
        )
