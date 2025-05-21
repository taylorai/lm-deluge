import time
from dataclasses import dataclass


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int | float = 0
    total_requests = 0

    @property
    def time_since_rate_limit_error(self):
        return time.time() - self.time_of_last_rate_limit_error

    def start_task(self, task_id):
        self.num_tasks_started += 1
        self.num_tasks_in_progress += 1

    def rate_limit_exceeded(self):
        self.time_of_last_rate_limit_error = time.time()
        self.num_rate_limit_errors += 1

    def task_succeeded(self, task_id):
        self.num_tasks_in_progress -= 1
        self.num_tasks_succeeded += 1

    def task_failed(self, task_id):
        self.num_tasks_in_progress -= 1
        self.num_tasks_failed += 1

    def log_final_status(self):
        if self.num_tasks_failed > 0:
            print(
                f"{self.num_tasks_failed} / {self.num_tasks_started} requests failed."
            )
        if self.num_rate_limit_errors > 0:
            print(
                f"{self.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )
