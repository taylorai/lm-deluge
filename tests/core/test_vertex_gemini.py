"""Offline Vertex Gemini request and registry checks."""

import asyncio
import os
from unittest.mock import patch

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.gemini import GeminiRequest, VertexGeminiRequest
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.models.google import GOOGLE_MODELS
from lm_deluge.prompt import Conversation
from lm_deluge.prompt.file import File


def context(model_name, prompt=None, sampling_params=None, **kwargs):
    return RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=prompt or Conversation().user("Hello"),
        sampling_params=sampling_params or SamplingParams(max_new_tokens=128),
        **kwargs,
    )


async def main():
    with patch.dict(os.environ, {"VERTEX_API_KEY": "secret", "GEMINI_API_KEY": "dev"}):
        for location, host in (
            ("global", "aiplatform.googleapis.com"),
            ("us", "aiplatform.us.rep.googleapis.com"),
            ("eu", "aiplatform.eu.rep.googleapis.com"),
            ("us-central1", "us-central1-aiplatform.googleapis.com"),
        ):
            with patch.dict(os.environ, {"VERTEX_LOCATION": location}):
                request = VertexGeminiRequest(context("gemini-3-flash-preview-vertex"))
                await request.build_request()
                assert request.url == (
                    f"https://{host}/v1beta1/publishers/google/models/"
                    "gemini-3-flash-preview:generateContent"
                )
                assert "secret" not in request.url
                assert request.request_header["x-goog-api-key"] == "secret"

        prompt = Conversation().user("Describe this", file=b"%PDF-1.4")
        vertex = VertexGeminiRequest(context("gemini-3-flash-preview-vertex", prompt))
        gemini = GeminiRequest(context("gemini-3-flash-preview", prompt))
        await vertex.build_request()
        await gemini.build_request()
        assert vertex.request_json == gemini.request_json
        assert "inlineData" in vertex.request_json["contents"][0]["parts"][1]

        poisoned = VertexGeminiRequest(
            context(
                "gemini-3-flash-preview-vertex",
                extra_headers={"X-Goog-Api-Key": "wrong", "X-Custom": "ok"},
            )
        )
        await poisoned.build_request()
        assert poisoned.request_header["x-goog-api-key"] == "secret"
        assert "X-Goog-Api-Key" not in poisoned.request_header
        assert poisoned.request_header["X-Custom"] == "ok"

        remote = File(
            data=None,
            is_remote=True,
            remote_provider="google",
            file_id="https://generativelanguage.googleapis.com/files/abc",
        )
        blocked = VertexGeminiRequest(
            context(
                "gemini-3-flash-preview-vertex",
                Conversation().user("Read", file=remote),
            )
        )
        try:
            await blocked.build_request()
        except ValueError as exc:
            assert "Gemini Files API" in str(exc)
        else:
            raise AssertionError("Gemini Files API upload was accepted")

    with patch.dict(os.environ, {"VERTEX_API_KEY": ""}):
        missing = VertexGeminiRequest(context("gemini-3-flash-preview-vertex"))
        try:
            await missing.build_request()
        except ValueError as exc:
            assert "VERTEX_API_KEY" in str(exc)
        else:
            raise AssertionError("missing Vertex API key was accepted")

    variants = [key for key in GOOGLE_MODELS if key.endswith("-vertex")]
    assert variants
    for variant in variants:
        sibling = variant.removesuffix("-vertex")
        vertex_model = APIModel.from_registry(variant)
        gemini_model = APIModel.from_registry(sibling)
        assert isinstance(
            vertex_model.make_request(context(variant)), VertexGeminiRequest
        )
        assert vertex_model.name == gemini_model.name
        for field in (
            "cached_input_cost",
            "reasoning_model",
            "omit_default_sampling_params",
            "input_cost",
            "cached_input_cost",
            "output_cost",
            "supports_images",
            "supports_json",
        ):
            assert getattr(vertex_model, field) == getattr(gemini_model, field)
    # Retired or Vertex-unavailable models must not get -vertex variants.
    for dead in ("gemini-3-pro-preview", "gemini-3.1-flash-lite-preview"):
        assert f"{dead}-vertex" not in GOOGLE_MODELS
    for added in ("gemini-3.1-pro-preview-vertex", "gemini-3.5-flash-lite-vertex"):
        assert added in GOOGLE_MODELS

    # 3.1 Pro accepts thinkingLevel=medium (live-probed 2026-10-02).
    pro = VertexGeminiRequest(
        context(
            "gemini-3.1-pro-preview-vertex",
            sampling_params=SamplingParams(
                max_new_tokens=128, reasoning_effort="medium"
            ),
        )
    )
    with patch.dict(os.environ, {"VERTEX_API_KEY": "secret"}):
        await pro.build_request()
    assert pro.request_json["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "medium"
    }
    print(f"PASS: Vertex Gemini offline checks ({len(variants)} registry variants)")


if __name__ == "__main__":
    asyncio.run(main())
