import asyncio
import json
import os
import tempfile
import time
from typing import Literal, Sequence, cast

import aiohttp
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, registry
from lm_deluge.prompt import (
    CachePattern,
    Conversation,
    Prompt,
    prompts_to_conversations,
)
from lm_deluge.request_context import RequestContext


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


async def submit_batch_oa(file_path: str):
    """Upload a JSONL file and create one OpenAI batch."""

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
        with open(file_path, "rb") as f:
            data.add_field(
                "file",
                f,
                filename=os.path.basename(file_path),
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

        os.remove(file_path)
        return batch_id


async def _submit_anthropic_batch(file_path: str, headers: dict, model: str):
    """Upload a JSONL file and create one Anthropic batch."""

    async with aiohttp.ClientSession() as session:
        url = f"{registry[model].api_base}/messages/batches"
        data = aiohttp.FormData()
        with open(file_path, "rb") as f:
            data.add_field(
                "file",
                f,
                filename=os.path.basename(file_path),
                content_type="application/json",
            )

            async with session.post(url, data=data, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ValueError(f"Error creating batch: {text}")

                batch_data = await response.json()
                batch_id = batch_data["id"]
                print(f"Anthropic batch job started successfully: id = {batch_id}")

        os.remove(file_path)
        return batch_id


async def create_batch_files_oa(
    model: str,
    sampling_params: SamplingParams,
    prompts: Prompt | Sequence[Prompt],
    batch_size: int = 50_000,
    destination: str | None = None,  # if none provided, temp files
):
    MAX_BATCH_SIZE_BYTES = 200 * 1024 * 1024  # 200MB
    MAX_BATCH_SIZE_ITEMS = batch_size

    if not isinstance(prompts, list):
        prompts = cast(Sequence[Prompt], [prompts])

    prompts = prompts_to_conversations(cast(Sequence[Prompt], prompts))
    assert isinstance(prompts, Sequence)
    if any(p is None for p in prompts):
        raise ValueError("All prompts must be valid.")

    model_obj = APIModel.from_registry(model)

    current_batch = []
    current_batch_size = 0
    file_paths = []

    for idx, prompt in enumerate(prompts):
        assert isinstance(prompt, Conversation)
        context = RequestContext(
            task_id=idx,
            model_name=model,
            prompt=prompt,
            sampling_params=sampling_params,
        )
        request = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": await _build_oa_chat_request(model_obj, context),
        }

        # Calculate size of this request
        request_json = json.dumps(request) + "\n"
        request_size = len(request_json.encode("utf-8"))

        # Check if adding this request would exceed limits
        would_exceed_size = current_batch_size + request_size > MAX_BATCH_SIZE_BYTES
        would_exceed_items = len(current_batch) >= MAX_BATCH_SIZE_ITEMS

        if current_batch and (would_exceed_size or would_exceed_items):
            # Submit current batch
            def write_batch_file():
                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".jsonl", delete=False
                ) as f:
                    for batch_request in current_batch:
                        json.dump(batch_request, f)
                        f.write("\n")
                    print("wrote", len(current_batch), "items")
                    return f.name

            file_path = await asyncio.to_thread(write_batch_file)
            file_paths.append(file_path)
            # Start new batch
            current_batch = []
            current_batch_size = 0
            # current_batch_start_idx = idx

        # Add request to current batch
        current_batch.append(request)
        current_batch_size += request_size

    # Submit final batch if it has items
    if current_batch:

        def write_final_batch_file():
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".jsonl", delete=False
            ) as f:
                for batch_request in current_batch:
                    json.dump(batch_request, f)
                    f.write("\n")
                print("wrote", len(current_batch), "items")
                return f.name

        file_path = await asyncio.to_thread(write_final_batch_file)
        file_paths.append(file_path)

    return file_paths


async def submit_batches_oa(
    model: str,
    sampling_params: SamplingParams,
    prompts: Prompt | Sequence[Prompt],
    batch_size: int = 50_000,
):
    """Write OpenAI batch requests to a file and submit."""
    MAX_BATCH_SIZE_BYTES = 200 * 1024 * 1024  # 200MB
    MAX_BATCH_SIZE_ITEMS = batch_size

    if not isinstance(prompts, list):
        prompts = prompts = cast(Sequence[Prompt], [prompts])

    prompts = prompts_to_conversations(cast(Sequence[Prompt], prompts))
    assert isinstance(prompts, Sequence)
    if any(p is None for p in prompts):
        raise ValueError("All prompts must be valid.")

    model_obj = APIModel.from_registry(model)

    tasks = []
    current_batch = []
    current_batch_size = 0
    # current_batch_start_idx = 0

    for idx, prompt in enumerate(prompts):
        assert isinstance(prompt, Conversation)
        context = RequestContext(
            task_id=idx,
            model_name=model,
            prompt=prompt,
            sampling_params=sampling_params,
        )
        request = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": await _build_oa_chat_request(model_obj, context),
        }

        # Calculate size of this request
        request_json = json.dumps(request) + "\n"
        request_size = len(request_json.encode("utf-8"))

        # Check if adding this request would exceed limits
        would_exceed_size = current_batch_size + request_size > MAX_BATCH_SIZE_BYTES
        would_exceed_items = len(current_batch) >= MAX_BATCH_SIZE_ITEMS

        if current_batch and (would_exceed_size or would_exceed_items):
            # Submit current batch
            def write_batch_file():
                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".jsonl", delete=False
                ) as f:
                    for batch_request in current_batch:
                        json.dump(batch_request, f)
                        f.write("\n")
                    print("wrote", len(current_batch), "items")
                    return f.name

            file_path = await asyncio.to_thread(write_batch_file)
            tasks.append(asyncio.create_task(submit_batch_oa(file_path)))

            # Start new batch
            current_batch = []
            current_batch_size = 0
            # current_batch_start_idx = idx

        # Add request to current batch
        current_batch.append(request)
        current_batch_size += request_size

    # Submit final batch if it has items
    if current_batch:

        def write_final_batch_file():
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".jsonl", delete=False
            ) as f:
                for batch_request in current_batch:
                    json.dump(batch_request, f)
                    f.write("\n")
                print("wrote", len(current_batch), "items")
                return f.name

        file_path = await asyncio.to_thread(write_final_batch_file)
        tasks.append(asyncio.create_task(submit_batch_oa(file_path)))

    batch_ids = await asyncio.gather(*tasks)

    print(f"Submitted {len(tasks)} batch jobs.")

    return batch_ids


async def submit_batches_anthropic(
    model: str,
    sampling_params: SamplingParams,
    prompts: Prompt | Sequence[Prompt],
    *,
    cache: CachePattern | None = None,
    batch_size=100_000,
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
    MAX_BATCH_SIZE_BYTES = 200 * 1024 * 1024  # 200MB
    MAX_BATCH_SIZE_ITEMS = batch_size

    # Convert prompts to Conversations
    if not isinstance(prompts, list):
        prompts = prompts = cast(Sequence[Prompt], [prompts])

    prompts = prompts_to_conversations(cast(Sequence[Prompt], prompts))

    request_headers = None
    batch_tasks = []
    current_batch = []
    current_batch_size = 0
    assert isinstance(prompts, Sequence)
    for idx, prompt in enumerate(prompts):
        assert isinstance(prompt, Conversation)
        context = RequestContext(
            task_id=idx,
            model_name=model,
            prompt=prompt,
            sampling_params=sampling_params,
            cache=cache,
        )
        request_body, request_headers = _build_anthropic_request(
            APIModel.from_registry(model), context
        )
        request = {"custom_id": str(idx), "params": request_body}

        # Calculate size of this request
        request_json = json.dumps(request) + "\n"
        request_size = len(request_json.encode("utf-8"))

        # Check if adding this request would exceed limits
        would_exceed_size = current_batch_size + request_size > MAX_BATCH_SIZE_BYTES
        would_exceed_items = len(current_batch) >= MAX_BATCH_SIZE_ITEMS

        if current_batch and (would_exceed_size or would_exceed_items):
            # Submit current batch
            def write_batch_file():
                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".jsonl", delete=False
                ) as f:
                    for batch_request in current_batch:
                        json.dump(batch_request, f)
                        f.write("\n")
                    print("wrote", len(current_batch), "items")
                    return f.name

            file_path = await asyncio.to_thread(write_batch_file)
            batch_tasks.append(
                asyncio.create_task(
                    _submit_anthropic_batch(file_path, request_headers, model)  # type: ignore
                )
            )

            # Start new batch
            current_batch = []
            current_batch_size = 0

        # Add request to current batch
        current_batch.append(request)
        current_batch_size += request_size

    # Submit final batch if it has items
    if current_batch:

        def write_final_batch_file():
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".jsonl", delete=False
            ) as f:
                for batch_request in current_batch:
                    json.dump(batch_request, f)
                    f.write("\n")
                print("wrote", len(current_batch), "items")
                return f.name

        file_path = await asyncio.to_thread(write_final_batch_file)
        batch_tasks.append(
            asyncio.create_task(
                _submit_anthropic_batch(file_path, request_headers, model)  # type: ignore
            )
        )

    batch_ids = await asyncio.gather(*batch_tasks)

    print(f"Submitted {len(batch_tasks)} batch jobs.")
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
