import os
import json
import time
import asyncio
import aiohttp
import pandas as pd
from lm_deluge.prompt import CachePattern, Conversation, prompts_to_conversations
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from typing import Sequence, Literal
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from lm_deluge.models import registry


def _create_batch_status_display(
    batch_id: str,
    status: str,
    elapsed: float,
    counts: dict | None,
    provider: str,
):
    """Create a unified status display for batch jobs."""
    # Format elapsed time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    if hours > 0:
        elapsed_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{seconds}s"

    # Build progress text based on provider
    progress_text = ""
    if counts:
        if provider == "openai":
            total = counts.get("total", 0)
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            total_display = "?" if total == 0 else str(total)
            progress_text = f" • {completed}/{total_display} done"
            if failed > 0:
                progress_text += f", {failed} failed"
        elif provider == "anthropic":
            total = (
                counts.get("processing", 0)
                + counts.get("succeeded", 0)
                + counts.get("errored", 0)
            )
            succeeded = counts.get("succeeded", 0)
            errored = counts.get("errored", 0)
            total_display = "?" if total == 0 else str(total)
            progress_text = f" • {succeeded}/{total_display} done"
            if errored > 0:
                progress_text += f", {errored} errors"

    # Choose spinner color based on provider
    spinner_style = "green" if provider == "openai" else "blue"
    spinner = Spinner("dots", style=spinner_style, text="")

    grid = Table.grid()
    grid.add_column()
    grid.add_column()
    grid.add_row(
        spinner,
        Text(
            f" Batch {batch_id} • {status} • {elapsed_str}{progress_text}",
            style="white",
        ),
    )
    return grid


async def submit_batch_oa(batch_requests: list[dict]):
    """Submit one batch asynchronously."""
    pd.DataFrame(batch_requests).to_json(
        "requests_temp.jsonl", orient="records", lines=True
    )

    # upload the file
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    async with aiohttp.ClientSession() as session:
        # Upload file
        url = "https://api.openai.com/v1/files"
        data = aiohttp.FormData()
        data.add_field("purpose", "batch")
        data.add_field(
            "file",
            open("requests_temp.jsonl", "rb"),
            filename="requests_temp.jsonl",
            content_type="application/json",
        )

        async with session.post(url, data=data, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error uploading file: {text}")

            print("File uploaded successfully")
            response_data = await response.json()
            file_id = response_data["id"]

        # Create batch
        url = "https://api.openai.com/v1/batches"
        batch_data = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }

        async with session.post(url, json=batch_data, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error starting batch job: {text}")

            response_data = await response.json()
            batch_id = response_data["id"]
            print("Batch job started successfully: id = ", batch_id)
            return batch_id


async def submit_batches_oa(
    model: str,
    sampling_params: SamplingParams,
    prompts: Sequence[str | list[dict] | Conversation],
):
    # if prompts are strings, convert them to message lists
    prompts = prompts_to_conversations(prompts)
    if any(p is None for p in prompts):
        raise ValueError("All prompts must be valid.")
    ids = [i for i, _ in enumerate(prompts)]

    # create file with requests to send to batch api
    batch_requests = []
    model_obj = APIModel.from_registry(model)
    for id, prompt in zip(ids, prompts):
        assert isinstance(prompt, Conversation)
        batch_requests.append(
            {
                "custom_id": str(id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _build_oa_chat_request(model_obj, prompt, [], sampling_params),
            }
        )

    # since the api only accepts up to 50,000 requests per batch job, we chunk into 50k chunks
    BATCH_SIZE = 50_000
    batches = [
        batch_requests[i : i + BATCH_SIZE]
        for i in range(0, len(batch_requests), BATCH_SIZE)
    ]
    tasks = []
    for batch in batches:
        tasks.append(asyncio.create_task(submit_batch_oa(batch)))
    batch_ids = await asyncio.gather(*tasks)

    print(f"Submitted {len(batches)} batch jobs.")

    return batch_ids


async def submit_batches_anthropic(
    model: str,
    sampling_params: SamplingParams,
    prompts: Sequence[str | list[dict] | Conversation],
    *,
    cache: CachePattern | None = None,
):
    """Submit a batch job to Anthropic's Message Batches API.

    Args:
        prompts: List of prompts to process
        wait_for_completion: If True, poll until completion and return results
        poll_interval: Seconds to wait between status checks when polling
        tools: Optional tools to include in requests
        cache: Optional cache pattern for requests

    Returns: batch_ids (list[str])
    """

    # Convert prompts to Conversations
    prompts = prompts_to_conversations(prompts)
    # Create batch requests
    request_headers = None
    batch_requests = []
    for i, prompt in enumerate(prompts):
        assert isinstance(prompt, Conversation)
        # Build request body
        request_body, request_headers = _build_anthropic_request(
            APIModel.from_registry(model), prompt, [], sampling_params, cache
        )

        batch_requests.append({"custom_id": str(i), "params": request_body})

    # Chunk into batches of 100k requests (Anthropic's limit)
    BATCH_SIZE = 100_000
    batches = [
        batch_requests[i : i + BATCH_SIZE]
        for i in range(0, len(batch_requests), BATCH_SIZE)
    ]
    batch_ids = []
    batch_tasks = []
    async with aiohttp.ClientSession() as session:
        for batch in batches:
            url = f"{registry[model]['api_base']}/messages/batches"
            data = {"requests": batch}

            async def submit_batch(data, url, headers):
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise ValueError(f"Error creating batch: {text}")

                    batch_data = await response.json()
                    batch_id = batch_data["id"]
                    print(f"Anthropic batch job started successfully: id = {batch_id}")
                    return batch_id

            batch_tasks.append(submit_batch(data, url, request_headers))

        batch_ids = await asyncio.gather(*batch_tasks)

    print(f"Submitted {len(batches)} batch jobs.")
    return batch_ids


async def wait_for_batch_completion_async(
    batch_ids: list[str],
    provider: Literal["openai", "anthropic"],
    poll_interval: int = 30,
):
    """Wait for multiple batches to complete and return results asynchronously.

    Args:
        batch_ids: List of batch IDs to wait for
        provider: Which provider the batches are from
        poll_interval: Seconds to wait between status checks

    Returns:
        List of results for each batch
    """
    tasks = []
    for batch_id in batch_ids:
        if provider == "openai":
            task = _wait_for_openai_batch_completion_async(batch_id, poll_interval)
        elif provider == "anthropic":
            task = _wait_for_anthropic_batch_completion_async(batch_id, poll_interval)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        tasks.append(task)

    # Wait for all batches concurrently
    results = await asyncio.gather(*tasks)

    results = [compl for batch in results for compl in batch]

    return results


async def _wait_for_anthropic_batch_completion_async(
    batch_id: str, poll_interval: int = 30
):
    """Poll Anthropic batch until completion and return results asynchronously."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
    console = Console()
    start_time = time.time()

    # Event to signal when to stop the display updater
    stop_display_event = asyncio.Event()
    current_status = {"status": "processing", "counts": None}

    async def display_updater():
        """Update display independently of polling."""
        with Live(console=console, refresh_per_second=10) as live:
            while not stop_display_event.is_set():
                elapsed = time.time() - start_time
                display = _create_batch_status_display(
                    batch_id,
                    current_status["status"],
                    elapsed,
                    current_status["counts"],
                    "anthropic",
                )
                live.update(display)
                await asyncio.sleep(0.1)  # Update every 100ms

    # Start display updater
    display_task = asyncio.create_task(display_updater())

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise ValueError(f"Error checking batch status: {text}")

                    batch_data = await response.json()
                    current_status["status"] = batch_data["processing_status"]
                    current_status["counts"] = batch_data.get("request_counts", {})

                    if current_status["status"] == "ended":
                        stop_display_event.set()
                        await display_task
                        console.print(
                            f"✅ Batch {batch_id} completed!", style="green bold"
                        )
                        return await _retrieve_anthropic_batch_results_async(batch_id)
                    elif current_status["status"] in ["canceled", "expired"]:
                        stop_display_event.set()
                        await display_task
                        raise ValueError(
                            f"Batch {batch_id} failed with status: {current_status['status']}"
                        )

                    await asyncio.sleep(poll_interval)
    finally:
        stop_display_event.set()
        await display_task


async def _retrieve_anthropic_batch_results_async(batch_id: str):
    """Retrieve results from completed Anthropic batch asynchronously."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error retrieving batch results: {text}")

            # Parse JSONL results
            results = []
            text = await response.text()
            for line in text.strip().split("\n"):
                if line:
                    result = json.loads(line)
                    results.append(result)

            # Sort by custom_id to maintain order
            results.sort(key=lambda x: int(x["custom_id"]))

            return results


async def _retrieve_openai_batch_results_async(batch_id: str):
    """Retrieve results from OpenAI batch asynchronously."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # Get batch info
        url = f"https://api.openai.com/v1/batches/{batch_id}"
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error retrieving batch: {text}")

            batch_data = await response.json()

            if batch_data["status"] != "completed":
                raise ValueError(
                    f"Batch {batch_id} is not completed. Status: {batch_data['status']}"
                )

            # Get output file
            output_file_id = batch_data["output_file_id"]
            if not output_file_id:
                raise ValueError(f"No output file available for batch {batch_id}")

        url = f"https://api.openai.com/v1/files/{output_file_id}/content"
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error retrieving batch results: {text}")

            # Parse JSONL results
            results = []
            text = await response.text()
            for line in text.strip().split("\n"):
                if line:
                    result = json.loads(line)
                    results.append(result)

            # Sort by custom_id to maintain order
            results.sort(key=lambda x: int(x["custom_id"]))

            return results


async def _wait_for_openai_batch_completion_async(
    batch_id: str, poll_interval: int = 30
):
    """Poll OpenAI batch until completion and return results asynchronously."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"https://api.openai.com/v1/batches/{batch_id}"
    console = Console()
    start_time = time.time()

    # Event to signal when to stop the display updater
    stop_display_event = asyncio.Event()
    current_status = {"status": "pending", "counts": None}

    async def display_updater():
        """Update display independently of polling."""
        with Live(console=console, refresh_per_second=10) as live:
            while not stop_display_event.is_set():
                elapsed = time.time() - start_time
                display = _create_batch_status_display(
                    batch_id,
                    current_status["status"],
                    elapsed,
                    current_status["counts"],
                    "openai",
                )
                live.update(display)
                await asyncio.sleep(0.1)  # Update every 100ms

    # Start display updater
    display_task = asyncio.create_task(display_updater())

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise ValueError(f"Error checking batch status: {text}")

                    batch_data = await response.json()
                    current_status["status"] = batch_data["status"]
                    current_status["counts"] = batch_data.get("request_counts", {})

                    if current_status["status"] == "completed":
                        stop_display_event.set()
                        await display_task
                        console.print(
                            f"✅ Batch {batch_id} completed!", style="green bold"
                        )
                        return await _retrieve_openai_batch_results_async(batch_id)
                    elif current_status["status"] in [
                        "failed",
                        "expired",
                        "cancelled",
                    ]:
                        stop_display_event.set()
                        await display_task
                        raise ValueError(
                            f"Batch {batch_id} failed with status: {current_status['status']}"
                        )

                    await asyncio.sleep(poll_interval)
    finally:
        stop_display_event.set()
        await display_task
