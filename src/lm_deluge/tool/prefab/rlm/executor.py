"""
RLM (Recursive Language Model) code executor.

Executes Python code with access to a context variable and lm() function
for recursive language model calls.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from .parse import (
    RLM_MODULES,
    RLM_SAFE_BUILTINS,
    RLMExecutionError,
    validate_rlm_code,
)

if TYPE_CHECKING:
    from lm_deluge.client import _LLMClient


class OutputCapture:
    """Captures print() output during execution."""

    def __init__(self):
        self.outputs: list[str] = []

    def print(self, *args, **kwargs):
        """Replacement print function that captures output."""
        output = " ".join(str(arg) for arg in args)
        self.outputs.append(output)

    def get_output(self) -> str:
        return "\n".join(self.outputs)


class PendingLMResult:
    """Placeholder for an lm() call result that hasn't completed yet."""

    def __init__(self, call_id: int, results: dict[int, str]):
        self._call_id = call_id
        self._results = results

    def _require_result(self) -> str:
        if self._call_id not in self._results:
            raise RuntimeError(f"LM result for call {self._call_id} not yet available")
        return self._results[self._call_id]

    def is_ready(self) -> bool:
        return self._call_id in self._results

    def __repr__(self) -> str:
        return repr(self._require_result())

    def __str__(self) -> str:
        return str(self._require_result())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._require_result(), name)

    def __getitem__(self, key: Any) -> Any:
        return self._require_result()[key]

    def __iter__(self):
        return iter(self._require_result())

    def __len__(self) -> int:
        return len(self._require_result())

    def __bool__(self) -> bool:
        return bool(self._require_result())

    def __add__(self, other):
        return self._require_result() + other

    def __radd__(self, other):
        return other + self._require_result()

    def __contains__(self, item):
        return item in self._require_result()


class FinalAnswer(Exception):
    """Raised when FINAL() or FINAL_VAR() is called to signal completion."""

    def __init__(self, answer: Any):
        self.answer = answer
        super().__init__("Final answer signaled")


def _resolve_value(value: Any, results: dict[int, str]) -> Any:
    """Recursively resolve any PendingLMResult placeholders in a value."""
    if isinstance(value, PendingLMResult):
        return value._require_result()
    if isinstance(value, list):
        return [_resolve_value(v, results) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(v, results) for v in value)
    if isinstance(value, dict):
        return {k: _resolve_value(v, results) for k, v in value.items()}
    if isinstance(value, set):
        return {_resolve_value(v, results) for v in value}
    return value


def _contains_unresolved(value: Any) -> bool:
    """Check if a value contains any unresolved PendingLMResult."""
    if isinstance(value, PendingLMResult):
        return not value.is_ready()
    if isinstance(value, (list, tuple, set)):
        return any(_contains_unresolved(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_unresolved(v) for v in value.values())
    return False


class RLMExecutor:
    """Executes RLM code with access to context and lm() calls."""

    def __init__(
        self,
        context: str,
        client: _LLMClient,
        context_var_name: str = "CONTEXT",
        max_lm_calls_per_execution: int = 20,
    ):
        """Initialize the RLM executor.

        Args:
            context: The long context string to analyze
            client: LLMClient for making recursive lm() calls
            context_var_name: Variable name for the context (default: "CONTEXT")
            max_lm_calls_per_execution: Maximum lm() calls allowed per execute() call
        """
        self.context = context
        self.client = client
        self.context_var_name = context_var_name
        self.max_lm_calls_per_execution = max_lm_calls_per_execution

        # Persistent state across execute() calls
        self._persistent_locals: dict[str, Any] = {}

    def _make_lm_wrapper(
        self,
        pending_lm_calls: list[dict],
        lm_results: dict[int, str],
        call_state: dict[str, int],
        pending_call_ids: set[int],
    ) -> Callable[[str], PendingLMResult]:
        """Create the lm(prompt) wrapper function."""

        def lm_call(prompt: str) -> PendingLMResult:
            # Check for unresolved dependencies in the prompt
            if _contains_unresolved(prompt):
                raise RuntimeError("LM result for call dependency not yet available")

            call_id = call_state["next_lm_id"]
            call_state["next_lm_id"] += 1

            # Only queue if not already completed or pending
            if call_id not in lm_results and call_id not in pending_call_ids:
                if len(pending_lm_calls) >= self.max_lm_calls_per_execution:
                    raise RuntimeError(
                        f"Too many lm() calls in single execution "
                        f"(max {self.max_lm_calls_per_execution})"
                    )
                pending_call_ids.add(call_id)
                pending_lm_calls.append(
                    {
                        "id": call_id,
                        "prompt": str(prompt),
                    }
                )

            return PendingLMResult(call_id, lm_results)

        return lm_call

    def _make_final_func(
        self, exec_namespace: dict[str, Any], lm_results: dict[int, str]
    ) -> Callable[[Any], None]:
        """Create final(answer) function."""

        def final_func(answer: Any) -> None:
            resolved = _resolve_value(answer, lm_results)
            raise FinalAnswer(resolved)

        return final_func

    def _make_final_var_func(
        self, exec_namespace: dict[str, Any], lm_results: dict[int, str]
    ) -> Callable[[str], None]:
        """Create final_var(varname) function."""

        def final_var_func(varname: str) -> None:
            if varname not in exec_namespace:
                raise RuntimeError(f"Variable '{varname}' not found")
            value = exec_namespace[varname]
            resolved = _resolve_value(value, lm_results)
            raise FinalAnswer(resolved)

        return final_var_func

    async def _execute_pending_lm_calls(
        self,
        pending_calls: list[dict],
        results: dict[int, str],
    ) -> None:
        """Execute all pending lm() calls in parallel."""
        if not pending_calls:
            return

        from lm_deluge.prompt import Conversation

        # Start all calls in parallel using start_nowait
        task_mapping: list[tuple[int, int]] = []  # (call_id, task_id)
        for call in pending_calls:
            conv = Conversation().user(call["prompt"])
            task_id = self.client.start_nowait(conv)
            task_mapping.append((call["id"], task_id))

        # Wait for all to complete
        for call_id, task_id in task_mapping:
            try:
                response = await self.client.wait_for(task_id)
                results[call_id] = response.completion or "(no response)"
            except Exception as e:
                results[call_id] = f"Error: {e}"

        # Clear the pending list
        pending_calls.clear()

    def _format_answer(self, value: Any) -> str:
        """Format the final answer as a string."""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, indent=2)
        except Exception:
            return str(value)

    async def execute(self, code: str) -> tuple[str, bool]:
        """Execute RLM code.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (output_string, is_final) where is_final indicates
            whether FINAL()/FINAL_VAR() was called.
        """
        # Validate the code
        tree = validate_rlm_code(code)

        # Set up execution environment
        pending_lm_calls: list[dict] = []
        lm_results: dict[int, str] = {}
        pending_call_ids: set[int] = set()
        call_state = {"next_lm_id": 0}
        output_capture = OutputCapture()

        # Create the lm() wrapper
        lm_wrapper = self._make_lm_wrapper(
            pending_lm_calls, lm_results, call_state, pending_call_ids
        )

        # Build a single namespace for execution
        # Using a single dict for both globals and locals ensures that
        # variables are visible inside nested scopes (list comprehensions, etc.)
        exec_namespace: dict[str, Any] = {
            "__builtins__": {**RLM_SAFE_BUILTINS, "print": output_capture.print},
            self.context_var_name: self.context,
            "lm": lm_wrapper,
            "json": json,  # Explicitly include json
            **RLM_MODULES,
            # Include persistent state from previous calls
            **self._persistent_locals,
        }

        # Add final and final_var (they need access to exec_namespace for final_var)
        exec_namespace["final"] = self._make_final_func(exec_namespace, lm_results)
        exec_namespace["final_var"] = self._make_final_var_func(
            exec_namespace, lm_results
        )

        # Track which keys are "system" keys that shouldn't be persisted
        system_keys = set(exec_namespace.keys())

        # Execute with retry loop for deferred lm() resolution
        max_iterations = 50
        compiled = compile(tree, "<rlm>", "exec")

        for iteration in range(max_iterations):
            # Reset call sequencing for this pass
            call_state["next_lm_id"] = 0
            pending_call_ids.clear()

            try:
                exec(compiled, exec_namespace)

                # Execution completed - run any remaining pending calls
                await self._execute_pending_lm_calls(pending_lm_calls, lm_results)

                # Update persistent locals (exclude system keys)
                for key, value in exec_namespace.items():
                    if key not in system_keys:
                        self._persistent_locals[key] = value

                break

            except FinalAnswer as fa:
                # FINAL() or FINAL_VAR() was called
                for key, value in exec_namespace.items():
                    if key not in system_keys:
                        self._persistent_locals[key] = value
                return (self._format_answer(fa.answer), True)

            except RuntimeError as e:
                if "not yet available" in str(e):
                    # Need to resolve pending lm() calls and retry
                    await self._execute_pending_lm_calls(pending_lm_calls, lm_results)
                    pending_call_ids.clear()
                    # Continue to retry
                else:
                    raise RLMExecutionError(f"Runtime error: {e}")

            except Exception as e:
                raise RLMExecutionError(f"Execution error: {type(e).__name__}: {e}")

        else:
            raise RLMExecutionError(
                f"Execution exceeded maximum iterations ({max_iterations})"
            )

        # Get output
        output = output_capture.get_output()

        # If no print output, check for result variable
        if not output and "result" in exec_namespace:
            result_value = _resolve_value(exec_namespace["result"], lm_results)
            output = self._format_answer(result_value)

        return (output or "Execution completed with no output", False)

    def reset(self) -> None:
        """Reset the persistent state."""
        self._persistent_locals.clear()
