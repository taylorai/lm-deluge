### specific utility for cohere rerank api
import asyncio
import os
import time
from dataclasses import dataclass

import aiohttp
from tqdm.auto import tqdm

from .tracker import StatusTracker

registry = [
    "rerank-english-v3.0",
    "rerank-multilingual-v3.0",
    "rerank-english-v2.0",
    "rerank-multilingual-v2.0",
]


class RerankingRequest:
    def __init__(
        self,
        task_id: int,
        model_name: str,
        query: str,
        documents: list[str],
        top_k: int,
        attempts_left: int,
        status_tracker: StatusTracker,
        request_timeout: int,
        pbar: tqdm | None = None,
    ):
        self.task_id = task_id
        self.model_name = model_name
        self.query = query
        self.documents = documents
        self.top_k = top_k
        self.attempts_left = attempts_left
        self.status_tracker = status_tracker
        self.request_timeout = request_timeout
        self.pbar = pbar
        self.result = []

    def increment_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def handle_success(self):
        self.increment_pbar()
        self.status_tracker.task_succeeded(self.task_id)

    def handle_error(self):
        """
        If create_new_request is True, will create a new API request (so that it
        has a chance of being sent to a different model). If false, will retry
        the same request.
        """
        last_result: RerankingResponse = self.result[-1]
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
                return RerankingResponse(
                    id=self.task_id,
                    status_code=response.status,
                    is_error=False,
                    error_message=None,
                    query=self.query,
                    documents=self.documents,
                    top_k_indices=[doc["index"] for doc in result["results"]],
                    top_k_scores=[doc["relevance_score"] for doc in result["results"]],
                )
            else:
                error_msg = await response.text()
                return RerankingResponse(
                    id=self.task_id,
                    status_code=response.status,
                    is_error=True,
                    error_message=error_msg,
                    query=self.query,
                    documents=[],
                    top_k_indices=[],
                    top_k_scores=[],
                )
        except Exception as e:
            return RerankingResponse(
                id=self.task_id,
                status_code=response.status,
                is_error=True,
                error_message=str(e),
                query=self.query,
                documents=[],
                top_k_indices=[],
                top_k_scores=[],
            )

    async def call_api(self):
        url = "https://api.cohere.com/v1/rerank"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {os.environ.get('COHERE_API_KEY')}",
        }
        data = {
            "model": self.model_name,
            "query": self.query,
            "top_n": self.top_k,
            "documents": self.documents,
        }
        try:
            self.status_tracker.total_requests += 1
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                async with session.post(
                    url, headers=headers, json=data, timeout=timeout
                ) as response:
                    # print("got response!!")
                    response_obj: RerankingResponse = await self.handle_response(
                        response
                    )
            self.result.append(response_obj)
            if response_obj.is_error:
                self.handle_error()
            else:
                self.handle_success()

        except asyncio.TimeoutError:
            self.result.append(
                RerankingResponse(
                    id=self.task_id,
                    status_code=None,
                    is_error=True,
                    error_message="Timeout",
                    query=self.query,
                    documents=[],
                    top_k_indices=[],
                    top_k_scores=[],
                )
            )
            self.handle_error()

        except Exception as e:
            self.result.append(
                RerankingResponse(
                    id=self.task_id,
                    status_code=None,
                    is_error=True,
                    error_message=f"Unexpected {type(e).__name__}: {str(e) or 'No message.'}",
                    query=self.query,
                    documents=[],
                    top_k_indices=[],
                    top_k_scores=[],
                )
            )
            self.handle_error()


@dataclass
class RerankingResponse:
    id: int
    status_code: int | None
    is_error: bool
    error_message: str | None
    query: str
    documents: list[str]
    top_k_indices: list[int]
    top_k_scores: list[float]

    @property
    def ranked_documents(self):
        return [self.documents[i] for i in self.top_k_indices]


async def rerank_parallel_async(
    queries: list[str],
    docs: list[list[str]],  # one list per query
    top_k: int = 3,
    model: str = "rerank-english-v3.0",
    max_attempts: int = 5,
    max_requests_per_minute: int = 4_000,
    max_concurrent_requests: int = 500,
    request_timeout: int = 10,
    progress_bar: tqdm | None = None,
):
    """Processes rerank requests in parallel, throttling to stay under rate limits."""
    ids = range(len(queries))
    # constants
    seconds_to_pause_after_rate_limit_error = 5
    seconds_to_sleep_each_loop = 0.003  # so concurrent tasks can run

    # initialize trackers
    # retry_queue = asyncio.Queue()
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
    prompts_iter = iter(zip(ids, queries, docs))
    results: list[RerankingRequest] = []

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            assert status_tracker.retry_queue

            if not status_tracker.retry_queue.empty():
                next_request = status_tracker.retry_queue.get_nowait()
                print(f"Retrying request {next_request.task_id}.")
            elif prompts_not_finished:
                try:
                    # get new request
                    req_id, req_query, req_docs = next(prompts_iter)
                    next_request = RerankingRequest(
                        task_id=req_id,
                        model_name=model,
                        query=req_query,
                        documents=req_docs,
                        top_k=top_k,
                        attempts_left=max_attempts,
                        status_tracker=status_tracker,
                        request_timeout=request_timeout,
                        pbar=progress_bar,
                    )
                    status_tracker.start_task(req_id)
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
        if progress_bar:
            if current_time - last_pbar_update_time > 1:
                last_pbar_update_time = current_time
                progress_bar.set_postfix(
                    {
                        "Request Capacity": f"{available_request_capacity:.1f}",
                        "Requests in Progress": status_tracker.num_tasks_in_progress,
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
                asyncio.create_task(next_request.call_api())
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
            current_response: RerankingResponse = deduplicated[request.task_id]
            # only replace if the current request has no top_k_indexes and the new one does
            if request.result[-1].top_k_indices and not current_response.top_k_indices:
                deduplicated[request.task_id] = request.result[-1]

    output = list(deduplicated.values())
    print(f"Returning {len(output)} unique results.")

    return output
