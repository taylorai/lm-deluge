import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
)
from tqdm.auto import tqdm

SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 5


@dataclass
class StatusTracker:
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_concurrent_requests: int
    client_name: str = "LLMClient"
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int | float = 0
    total_requests: int = 0
    retry_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Cumulative usage tracking
    total_cost: float = 0.0
    total_input_tokens: int = 0  # non-cached input tokens
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_output_tokens: int = 0

    # Progress bar configuration
    use_progress_bar: bool = True
    progress_bar_total: int | None = None
    progress_bar_disable: bool = False
    progress_style: Literal["rich", "tqdm", "manual"] = "rich"
    progress_print_interval: float = 30.0
    _pbar: tqdm | None = None

    # Rich display configuration
    _rich_console: Console | None = None
    _rich_live: object | None = None
    _rich_progress: Progress | None = None
    _rich_task_id: TaskID | None = None
    _rich_display_task: asyncio.Task | None = None
    _rich_stop_event: asyncio.Event | None = None

    # Manual print configuration
    _manual_display_task: asyncio.Task | None = None
    _manual_stop_event: asyncio.Event | None = None

    def __post_init__(self):
        self.available_request_capacity = self.max_requests_per_minute
        self.available_token_capacity = self.max_tokens_per_minute
        self.last_update_time = time.time() - 1
        self.last_pbar_update_time = time.time() - 1
        self.limiting_factor = None

    @property
    def time_since_rate_limit_error(self):
        return time.time() - self.time_of_last_rate_limit_error

    @property
    def seconds_to_pause(self):
        return max(
            0,
            SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR - self.time_since_rate_limit_error,
        )

    def set_limiting_factor(self, factor):
        self.limiting_factor = factor

    def check_capacity(self, num_tokens: int, retry: bool = False):
        self.update_capacity()  # always update before checking
        request_available = self.available_request_capacity >= 1
        tokens_available = self.available_token_capacity >= num_tokens
        concurrent_request_available = (
            self.num_tasks_in_progress < self.max_concurrent_requests
        )
        if request_available and tokens_available and concurrent_request_available:
            self.available_request_capacity -= 1
            self.available_token_capacity -= num_tokens
            if not retry:
                # Only count new tasks, not retries
                self.num_tasks_started += 1
                self.num_tasks_in_progress += 1
            self.set_limiting_factor(None)
            return True
        else:
            # update reason why
            if not request_available:
                self.set_limiting_factor("Requests")
            elif not concurrent_request_available:
                self.set_limiting_factor("Concurrent Requests")
            elif not tokens_available:
                self.set_limiting_factor("Tokens")

    def update_capacity(self):
        current_time = time.time()
        seconds_since_update = current_time - self.last_update_time
        self.available_request_capacity = min(
            self.available_request_capacity
            + self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute,
        )
        self.available_token_capacity = min(
            self.available_token_capacity
            + self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute,
        )
        self.last_update_time = current_time

    def start_task(self, task_id):
        self.num_tasks_started += 1
        self.num_tasks_in_progress += 1

    def rate_limit_exceeded(self):
        self.time_of_last_rate_limit_error = time.time()
        self.num_rate_limit_errors += 1

    def task_succeeded(self, task_id):
        self.num_tasks_in_progress -= 1
        self.num_tasks_succeeded += 1
        self.increment_pbar()

    def task_failed(self, task_id):
        self.num_tasks_in_progress -= 1
        self.num_tasks_failed += 1

    def track_usage(self, response):
        """Accumulate usage statistics from a completed request.

        Args:
            response: APIResponse object containing usage and cost information
        """
        if response.cost:
            self.total_cost += response.cost

        if response.usage:
            self.total_output_tokens += response.usage.output_tokens
            self.total_input_tokens += response.usage.input_tokens

            if response.usage.cache_read_tokens:
                self.total_cache_read_tokens += response.usage.cache_read_tokens

            if response.usage.cache_write_tokens:
                self.total_cache_write_tokens += response.usage.cache_write_tokens

    def log_final_status(self):
        # Close progress bar before printing final status
        self.close_progress_bar()

        if self.num_tasks_failed > 0:
            print(
                f"{self.num_tasks_failed} / {self.num_tasks_started} requests failed."
            )
        if self.num_rate_limit_errors > 0:
            print(
                f"{self.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

        # Display cumulative usage stats if available
        if (
            self.total_cost > 0
            or self.total_input_tokens > 0
            or self.total_output_tokens > 0
        ):
            usage_parts = []
            if self.total_cost > 0:
                usage_parts.append(f"💰 Cost: ${self.total_cost:.4f}")
            if self.total_input_tokens > 0 or self.total_output_tokens > 0:
                usage_parts.append(
                    f"🔡 Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
                )
            if self.total_cache_read_tokens > 0:
                usage_parts.append(f"Cache: {self.total_cache_read_tokens:,} read")
            if self.total_cache_write_tokens > 0:
                usage_parts.append(f"{self.total_cache_write_tokens:,} write")

            print("  ", " • ".join(usage_parts))

    @property
    def pbar(self) -> tqdm | None:
        """Backward compatibility property to access progress bar."""
        return self._pbar

    def init_progress_bar(self, total: int | None = None, disable: bool | None = None):
        """Initialize progress bar if enabled."""
        if not self.use_progress_bar:
            return

        pbar_total = total if total is not None else self.progress_bar_total
        pbar_disable = disable if disable is not None else self.progress_bar_disable
        if pbar_total is None:
            pbar_total = 0
        self.progress_bar_total = pbar_total

        if self.progress_style == "rich":
            if pbar_disable:
                return
            self._init_rich_display(pbar_total)
        elif self.progress_style == "tqdm":
            self._pbar = tqdm(total=pbar_total, disable=pbar_disable)
        elif self.progress_style == "manual":
            self._init_manual_display(pbar_total)

        self.update_pbar()

    def close_progress_bar(self):
        """Close progress bar if it exists."""
        if not self.use_progress_bar:
            return
        if self.progress_style == "rich":
            if self._rich_stop_event:
                self._close_rich_display()
        elif self.progress_style == "tqdm":
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
        elif self.progress_style == "manual":
            self._close_manual_display()

    def _init_rich_display(self, total: int):
        """Initialize Rich display components."""
        self._rich_console = Console(highlight=False)
        # Escape square brackets so Rich doesn't interpret them as markup
        description = f"[bold blue]\\[{self.client_name}][/bold blue] Processing..."
        self._rich_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        )
        self._rich_task_id = self._rich_progress.add_task(description, total=total)
        self._rich_stop_event = asyncio.Event()
        self._rich_display_task = asyncio.create_task(self._rich_display_updater())

    async def _rich_display_updater(self):
        """Update Rich display independently."""
        if (
            not self._rich_console
            or self._rich_progress is None
            or self._rich_task_id is None
            or self._rich_stop_event is None
        ):
            return

        with Live(console=self._rich_console, refresh_per_second=10) as live:
            while not self._rich_stop_event.is_set():
                completed = self.num_tasks_succeeded
                self._rich_progress.update(
                    self._rich_task_id,
                    completed=completed,
                    total=self.progress_bar_total,
                )

                tokens_info = f"{self.available_token_capacity / 1000:.1f}k/{self.max_tokens_per_minute / 1000:.1f}k TPM"
                reqs_info = f"{int(self.available_request_capacity)}/{self.max_requests_per_minute} RPM"
                in_progress = (
                    f"   [gold3]In Progress:[/gold3] {int(self.num_tasks_in_progress)} "
                    + ("requests" if self.num_tasks_in_progress != 1 else "request")
                )
                capacity_text = (
                    f"   [gold3]Capacity:[/gold3] {tokens_info} • {reqs_info}"
                )

                # Format usage stats
                usage_parts = []
                if self.total_cost > 0:
                    usage_parts.append(f"${self.total_cost:.4f}")
                if self.total_input_tokens > 0 or self.total_output_tokens > 0:
                    input_k = self.total_input_tokens / 1000
                    output_k = self.total_output_tokens / 1000
                    usage_parts.append(f"{input_k:.1f}k in • {output_k:.1f}k out")
                if self.total_cache_read_tokens > 0:
                    cache_k = self.total_cache_read_tokens / 1000
                    usage_parts.append(f"{cache_k:.1f}k cached")

                usage_text = ""
                if usage_parts:
                    usage_text = f"   [gold3]Usage:[/gold3] {' • '.join(usage_parts)}"

                if usage_text:
                    display = Group(
                        self._rich_progress, in_progress, capacity_text, usage_text
                    )
                else:
                    display = Group(self._rich_progress, in_progress, capacity_text)
                live.update(display)

                await asyncio.sleep(0.1)

    def _close_rich_display(self):
        """Clean up Rich display."""
        if self._rich_stop_event:
            self._rich_stop_event.set()
        if self._rich_display_task and not self._rich_display_task.done():
            self._rich_display_task.cancel()

        self._rich_console = None
        self._rich_live = None
        self._rich_progress = None
        self._rich_task_id = None
        self._rich_display_task = None
        self._rich_stop_event = None

    def _init_manual_display(self, total: int):
        """Initialize manual progress printer."""
        self.progress_bar_total = total
        self._manual_stop_event = asyncio.Event()
        self._manual_display_task = asyncio.create_task(self._manual_display_updater())

    async def _manual_display_updater(self):
        if self._manual_stop_event is None:
            return
        while not self._manual_stop_event.is_set():
            print(
                f"[{self.client_name}] Completed {self.num_tasks_succeeded}/{self.progress_bar_total} requests"
            )
            await asyncio.sleep(self.progress_print_interval)

    def _close_manual_display(self):
        if self._manual_stop_event:
            self._manual_stop_event.set()
        if self._manual_display_task and not self._manual_display_task.done():
            self._manual_display_task.cancel()
        self._manual_display_task = None
        self._manual_stop_event = None

    def update_pbar(self, n: int = 0):
        """Update progress bar status and optionally increment.

        Args:
            n: Number of items to increment (0 means just update postfix)
        """
        if self.progress_style != "tqdm":
            return

        current_time = time.time()
        if self._pbar and (current_time - self.last_pbar_update_time > 1):
            self.last_pbar_update_time = current_time
            self._pbar.set_postfix(
                {
                    "Token Capacity": f"{self.available_token_capacity / 1_000:.1f}k",
                    "Req. Capacity": f"{int(self.available_request_capacity)}",
                    "Reqs. in Progress": self.num_tasks_in_progress,
                    "Limiting Factor": self.limiting_factor,
                }
            )

        if n > 0 and self._pbar:
            self._pbar.update(n)

    def increment_pbar(self):
        """Increment progress bar by 1."""
        if not self.use_progress_bar:
            return
        if self.progress_style == "tqdm" and self._pbar:
            self._pbar.update(1)
        # rich and manual are updated elsewhere

    def add_to_total(self, n: int = 1):
        """Increase the total number of tasks being tracked."""
        if self.progress_bar_total is None:
            self.progress_bar_total = 0
        self.progress_bar_total += n
        if not self.use_progress_bar:
            return
        if self.progress_style == "tqdm" and self._pbar:
            self._pbar.total = self.progress_bar_total
            self._pbar.refresh()
        elif (
            self.progress_style == "rich"
            and self._rich_progress
            and self._rich_task_id is not None
        ):
            self._rich_progress.update(
                self._rich_task_id, total=self.progress_bar_total
            )
