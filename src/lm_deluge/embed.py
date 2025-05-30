### specific utility for cohere rerank api
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np
from tqdm.auto import tqdm

from .tracker import StatusTracker

registry = {
    "text-embedding-3-small": {
        "name": "text-embedding-3-small",
        "provider": "openai",
        "cost": 0.02,  # per million tokens
    },
    "text-embedding-3-large": {
        "name": "text-embedding-3-large",
        "provider": "openai",
        "cost": 0.13,
    },
    "text-embedding-ada-002": {
        "name": "text-embedding-ada-002",
        "provider": "openai",
        "cost": 1,
    },
    "embed-english-v3.0": {
        "name": "embed-english-v3.0",
        "provider": "cohere",
        "cost": 0.1,
    },
    "embed-english-light-v3.0": {
        "name": "embed-english-light-v3.0",
        "provider": "cohere",
        "cost": 0.1,
    },
    "embed-multilingual-v3.0": {
        "name": "embed-multilingual-v3.0",
        "provider": "cohere",
        "cost": 0.1,
    },
    "embed-multilingual-light-v3.0": {
        "name": "embed-multilingual-light-v3.0",
        "provider": "cohere",
        "cost": 0.1,
    },
}


class EmbeddingRequest:
    def __init__(
        self,
        task_id: int,
        model_name: str,
        texts: list[str],
        attempts_left: int,
        status_tracker: StatusTracker,
        request_timeout: int,
        pbar: tqdm | None = None,
        **kwargs,  # openai or cohere specific params
    ):
        self.task_id = task_id
        self.model_name = model_name
        self.texts = texts
        self.attempts_left = attempts_left
        self.status_tracker = status_tracker
        self.request_timeout = request_timeout
        self.pbar = pbar
        self.result = []
        self.kwargs = kwargs

    def increment_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def handle_success(self):
        self.increment_pbar()
        self.status_tracker.task_succeeded(self.task_id)

    def handle_error(self):
        last_result: EmbeddingResponse = self.result[-1]
        error_to_print = (
            f"Error on task {self.task_id}, Code: {last_result.status_code}, "
        )
        error_to_print += f"Message: {last_result.error_message}."
        print(error_to_print)
        if self.attempts_left > 0:
            self.attempts_left -= 1
            assert self.status_tracker.retry_queue
            self.status_tracker.retry_queue.put_nowait(self)
            return
        else:
            print(f"Task {self.task_id} out of tries.")
            self.status_tracker.task_failed(self.task_id)

    async def handle_response(self, response: aiohttp.ClientResponse):
        try:
            if response.status == 200:
                result = await response.json()
                # TODO: add cost calculation
                if self.model_name in [
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002",
                ]:
                    embeddings = [
                        embedding["embedding"] for embedding in result["data"]
                    ]
                elif self.model_name in [
                    "embed-english-v3.0",
                    "embed-english-light-v3.0",
                    "embed-multilingual-v3.0",
                    "embed-multilingual-light-v3.0",
                ]:
                    embeddings = result["embeddings"]
                else:
                    raise ValueError(f"Unsupported model {self.model_name}")
                return EmbeddingResponse(
                    id=self.task_id,
                    status_code=response.status,
                    is_error=False,
                    error_message=None,
                    texts=self.texts,
                    embeddings=embeddings,
                )
            else:
                error_msg = await response.text()
                return EmbeddingResponse(
                    id=self.task_id,
                    status_code=response.status,
                    is_error=True,
                    error_message=error_msg,
                    texts=[],
                    embeddings=[],
                )
        except Exception as e:
            return EmbeddingResponse(
                id=self.task_id,
                status_code=response.status,
                is_error=True,
                error_message=str(e),
                texts=[],
                embeddings=[],
            )

    async def call_api(
        self,
        session: aiohttp.ClientSession,
    ):
        if len(self.texts) > 96:
            raise ValueError("Embeddings only support up to 96 texts per request.")
        model_obj = registry[self.model_name]
        url = (
            "https://api.openai.com/v1/embeddings"
            if model_obj["provider"] == "openai"
            else "https://api.cohere.com/v1/embed"
        )
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            if model_obj["provider"] == "openai"
            else f"bearer {os.environ.get('COHERE_API_KEY')}"
        }
        payload: dict[str, Any] = {"model": self.model_name}
        if model_obj["provider"] == "openai":
            payload["input"] = self.texts
            payload["encoding_format"] = "float"
            for k, v in self.kwargs.items():
                payload[k] = v
        elif model_obj["provider"] == "cohere":
            payload["texts"] = self.texts
            payload["input_type"] = self.kwargs.get("input_type", "search_document")
            for k, v in self.kwargs.items():
                payload[k] = v
        try:
            self.status_tracker.total_requests += 1
            async with session.post(url, json=payload, headers=headers) as response:
                response_obj: EmbeddingResponse = await self.handle_response(response)
            self.result.append(response_obj)
            if response_obj.is_error:
                self.handle_error()
            else:
                self.handle_success()

        except asyncio.TimeoutError:
            self.result.append(
                EmbeddingResponse(
                    id=self.task_id,
                    status_code=None,
                    is_error=True,
                    error_message="Timeout",
                    texts=[],
                    embeddings=[],
                )
            )
            self.handle_error()

        except Exception as e:
            self.result.append(
                EmbeddingResponse(
                    id=self.task_id,
                    status_code=None,
                    is_error=True,
                    error_message=f"Unexpected {type(e).__name__}: {str(e) or 'No message.'}",
                    texts=[],
                    embeddings=[],
                )
            )
            self.handle_error()


@dataclass
class EmbeddingResponse:
    id: int
    status_code: int | None
    is_error: bool
    error_message: str | None
    texts: list[str]
    embeddings: list[list[float]]


async def embed_parallel_async(
    texts: list[str],
    model: str = "rerank-english-v3.0",
    max_attempts: int = 5,
    max_requests_per_minute: int = 4_000,
    max_concurrent_requests: int = 500,
    request_timeout: int = 10,
    batch_size: int = 16,
    show_progress: bool = True,
    **kwargs,
):
    """Processes embed requests in parallel, throttling to stay under rate limits."""
    if batch_size > 96:
        raise ValueError("Embeddings only support up to 96 texts per request.")
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    pbar = tqdm(total=len(batches), desc="Embedding") if show_progress else None
    ids = range(len(batches))
    # constants
    seconds_to_pause_after_rate_limit_error = 5
    seconds_to_sleep_each_loop = 0.003  # so concurrent tasks can run

    # initialize trackers
    retry_queue = asyncio.Queue()
    status_tracker = StatusTracker(
        max_tokens_per_minute=10_000_000,
        max_requests_per_minute=max_requests_per_minute,
        max_concurrent_requests=1_000,
    )
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    # throttle over a 1 second window rather than minute,
    # since some models limit RPS rather than RPM
    available_request_capacity = max_requests_per_minute
    last_update_time = time.time()
    last_pbar_update_time = time.time()

    # initialize flags
    prompts_not_finished = True
    prompts_iter = iter(zip(ids, batches))
    results: list = []
    session = aiohttp.ClientSession()

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            assert status_tracker.retry_queue
            if not status_tracker.retry_queue.empty():
                next_request = retry_queue.get_nowait()
                print(f"Retrying request {next_request.task_id}.")
            elif prompts_not_finished:
                try:
                    # get new request
                    batch_id, batch = next(prompts_iter)
                    next_request = EmbeddingRequest(
                        task_id=batch_id,
                        model_name=model,
                        texts=batch,
                        attempts_left=max_attempts,
                        status_tracker=status_tracker,
                        retry_queue=retry_queue,
                        request_timeout=request_timeout,
                        pbar=pbar,
                        **kwargs,
                    )
                    status_tracker.start_task(batch_id)
                    results.append(next_request)

                except StopIteration:
                    prompts_not_finished = False
                    # print("API requests finished, only retries remain.")

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        last_update_time = current_time

        # update pbar status
        if pbar:
            if current_time - last_pbar_update_time > 1:
                last_pbar_update_time = current_time
                pbar.set_postfix(
                    {
                        "Req. Capacity": f"{available_request_capacity:.1f}",
                        "Reqs in Progress": status_tracker.num_tasks_in_progress,
                    }
                )

        # if enough capacity available, call API
        if next_request:
            if (
                available_request_capacity >= 1
                and status_tracker.num_tasks_in_progress < max_concurrent_requests
            ):
                # update counters
                available_request_capacity -= 1
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(next_request.call_api(session=session))
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        remaining_seconds_to_pause = max(
            0,
            seconds_to_pause_after_rate_limit_error
            - status_tracker.time_since_rate_limit_error,
        )
        if remaining_seconds_to_pause > 0:
            await asyncio.sleep(remaining_seconds_to_pause)
            print(f"Pausing {remaining_seconds_to_pause}s to cool down.")

    # after finishing, log final status
    status_tracker.log_final_status()
    print(
        f"After processing, got {len(results)} results for {len(ids)} inputs. Removing duplicates."
    )

    # deduplicate results by id
    deduplicated = {}
    for request in results:
        if request.task_id not in deduplicated:
            deduplicated[request.task_id] = request.result[-1]
        else:
            current_response: EmbeddingResponse = deduplicated[request.task_id]
            # only replace if the current request has no top_k_indexes and the new one does
            if request.result[-1].embeddings and not current_response.embeddings:
                deduplicated[request.task_id] = request.result[-1]

    output = list(deduplicated.values())
    # sort by id
    output.sort(key=lambda x: x.id)
    print(f"Returning {len(output)} unique results.")
    await session.close()
    return output


def stack_results(
    results: list[EmbeddingResponse], return_numpy: bool = True
) -> list[list[float]] | np.ndarray:
    if not all(response.status_code == 200 for response in results):
        raise ValueError("Some responses were not successful; cannot coalesce results.")
    stacked = np.concatenate([response.embeddings for response in results], axis=0)
    return stacked.tolist() if not return_numpy else stacked  # type: ignore


def submit_batch_request():
    pass
