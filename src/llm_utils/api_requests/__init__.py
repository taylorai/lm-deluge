import asyncio
from tqdm import tqdm
from .anthropic import AnthropicRequest
from .vertex import VertexAnthropicRequest, GeminiRequest
from .bedrock import BedrockAnthropicRequest
from .openai import OpenAIRequest
from .cohere import CohereRequest
from ..tracker import StatusTracker
from ..cache import SqliteCache
from ..sampling_params import SamplingParams
from ..models import APIModel

from typing import Optional, Callable

def create_api_request(
    task_id: int,
    model_name: str,
    messages: list[dict], 
    attempts_left: int,
    status_tracker: StatusTracker,
    retry_queue: asyncio.Queue,
    request_timeout: int = 30,
    sampling_params: SamplingParams = SamplingParams(),
    cache: Optional[SqliteCache] = None,
    pbar: Optional[tqdm] = None,
    callback: Optional[Callable] = None
):
    model_obj = APIModel.from_registry(model_name)
    if model_obj.api_spec == "openai":
        return OpenAIRequest(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback
        )
    elif model_obj.api_spec == "anthropic":
        return AnthropicRequest(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback
        )
    elif model_obj.api_spec == "vertex_anthropic":
        return VertexAnthropicRequest(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback
        )
    elif model_obj.api_spec == "vertex_gemini":
        return GeminiRequest(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback
        )
    elif model_obj.api_spec == "cohere":
        return CohereRequest(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback
        )
    elif model_obj.api_spec == "bedrock_anthropic":
        return BedrockAnthropicRequest(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback
        )

    else:
        raise ValueError(f"Unsupported API spec: {model_obj.api_spec}")