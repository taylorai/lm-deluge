from pathlib import Path
from typing import Any, Literal

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.client import LLMClient
from lm_deluge.prompt import Conversation, Image
from lm_deluge.util.spatial import SpatialResult, parse_moondream


ImageInput = Image | bytes | str | Path
ProgressStyle = Literal["rich", "tqdm", "manual"]


class MoondreamClient:
    def __init__(
        self,
        model_name: str = "moondream",
        *,
        max_attempts: int = 5,
        request_timeout: int = 30,
        max_new_tokens: int = 512,
        progress: ProgressStyle = "rich",
        **client_kwargs: Any,
    ):
        self.model_name: str = model_name
        self.max_attempts: int = max_attempts
        self.request_timeout: int = request_timeout
        self.max_new_tokens: int = max_new_tokens
        self.progress: ProgressStyle = progress
        self.client_kwargs: dict[str, Any] = client_kwargs

    async def query(
        self,
        image: ImageInput,
        question: str,
        *,
        reasoning: bool = False,
    ) -> APIResponse:
        return await self._client(reasoning=reasoning).start(
            Conversation().user(question, image=_coerce_image(image))
        )

    async def caption(
        self,
        image: ImageInput,
        *,
        length: str = "normal",
    ) -> APIResponse:
        return await self._client(task="caption", length=length).start(
            Conversation().user("", image=_coerce_image(image))
        )

    async def detect(
        self,
        image: ImageInput,
        target: str,
        *,
        max_objects: int | None = None,
    ) -> SpatialResult:
        response = await self._client(
            task="detect", object=target, max_objects=max_objects
        ).start(Conversation().user("", image=_coerce_image(image)))
        _raise_for_error(response)
        assert response.raw_response is not None
        return parse_moondream(response.raw_response)

    async def point(
        self,
        image: ImageInput,
        target: str,
        *,
        max_objects: int | None = None,
    ) -> SpatialResult:
        response = await self._client(
            task="point", object=target, max_objects=max_objects
        ).start(Conversation().user("", image=_coerce_image(image)))
        _raise_for_error(response)
        assert response.raw_response is not None
        return parse_moondream(response.raw_response)

    async def segment(
        self,
        image: ImageInput,
        target: str,
        *,
        spatial_refs: list | None = None,
    ) -> SpatialResult:
        response = await self._client(
            task="segment", object=target, spatial_refs=spatial_refs
        ).start(Conversation().user("", image=_coerce_image(image)))
        _raise_for_error(response)
        assert response.raw_response is not None
        return parse_moondream(response.raw_response)

    def _client(self, **extra_body: Any):
        clean_extra = {
            key: value for key, value in extra_body.items() if value is not None
        }
        return LLMClient(
            self.model_name,
            max_attempts=self.max_attempts,
            request_timeout=self.request_timeout,
            max_new_tokens=self.max_new_tokens,
            progress=self.progress,
            extra_body=clean_extra or None,
            **self.client_kwargs,
        )


def _coerce_image(image: ImageInput) -> Image:
    if isinstance(image, Image):
        return image
    return Image(image)


def _raise_for_error(response: APIResponse) -> None:
    if response.is_error:
        raise RuntimeError(response.error_message or "Moondream request failed")
