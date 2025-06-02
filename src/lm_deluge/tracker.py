import asyncio
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.text import Text
from tqdm import tqdm

SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 5


@dataclass
class StatusTracker:
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_concurrent_requests: int
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int | float = 0
    total_requests: int = 0
    retry_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Progress bar configuration
    use_progress_bar: bool = True
    progress_bar_total: int | None = None
    progress_bar_disable: bool = False
    _pbar: tqdm | None = None

    # Rich display configuration
    use_rich: bool = True
    _rich_console: Console | None = None
    _rich_live: object | None = None
    _rich_progress: object | None = None
    _rich_task_id: object | None = None
    _rich_display_task: asyncio.Task | None = None
    _rich_stop_event: asyncio.Event | None = None

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

    @property
    def pbar(self) -> tqdm | None:
        """Backward compatibility property to access progress bar."""
        return self._pbar

    def init_progress_bar(self, total: int | None = None, disable: bool | None = None):
        """Initialize progress bar if enabled."""
        if not self.use_progress_bar:
            return

        if self.use_rich:
            self._init_rich_display(total, disable)
        else:
            # Use provided values or fall back to instance defaults
            pbar_total = total if total is not None else self.progress_bar_total
            pbar_disable = disable if disable is not None else self.progress_bar_disable
            self._pbar = tqdm(total=pbar_total, disable=pbar_disable)
        self.update_pbar()

    def close_progress_bar(self):
        """Close progress bar if it exists."""
        if self.use_rich and self._rich_stop_event:
            self._close_rich_display()
        elif self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _init_rich_display(self, total: int | None = None, disable: bool | None = None):
        """Initialize Rich display components."""
        if disable:
            return

        pbar_total = total if total is not None else self.progress_bar_total
        if pbar_total is None:
            pbar_total = 100  # Default fallback

        self._rich_console = Console()
        self._rich_stop_event = asyncio.Event()

        # Start the display updater task
        self._rich_display_task = asyncio.create_task(
            self._rich_display_updater(pbar_total)
        )

    async def _rich_display_updater(self, total: int):
        """Update Rich display independently."""
        if not self._rich_console or self._rich_stop_event is None:
            return

        # Create progress bar without console so we can use it in Live
        progress = Progress(
            SpinnerColumn(),
            TextColumn("Processing requests..."),
            BarColumn(),
            MofNCompleteColumn(),
        )
        main_task = progress.add_task("requests", total=total)

        # Use Live to combine progress + text

        with Live(console=self._rich_console, refresh_per_second=10) as live:
            while not self._rich_stop_event.is_set():
                completed = self.num_tasks_succeeded
                progress.update(main_task, completed=completed)

                # Create capacity info text
                tokens_info = f"TPM Capacity: {self.available_token_capacity / 1000:.1f}k/{self.max_tokens_per_minute / 1000:.1f}k"
                reqs_info = f"RPM Capacity: {int(self.available_request_capacity)}/{self.max_requests_per_minute}"
                in_progress = f"In Progress: {int(self.num_tasks_in_progress)}"
                capacity_text = Text(f"{in_progress} • {tokens_info} • {reqs_info}")

                # Group progress bar and text
                display = Group(progress, capacity_text)
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
        self._rich_display_task = None
        self._rich_stop_event = None

    def update_pbar(self, n: int = 0):
        """Update progress bar status and optionally increment.

        Args:
            n: Number of items to increment (0 means just update postfix)
        """
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
        if self.use_rich:
            # Rich display is updated automatically by the display updater
            pass
        elif self._pbar:
            self._pbar.update(1)
