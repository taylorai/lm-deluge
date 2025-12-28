import asyncio
import json
from typing import Any, Callable

from lm_deluge.tool import Tool

from .parse import SAFE_BUILTINS, OTCExecutionError, validate_code


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


class PendingResult:
    """Placeholder for a tool call result."""

    def __init__(self, call_id: int, results: dict[int, Any]):
        self._call_id = call_id
        self._results = results

    def _require_result(self) -> Any:
        if self._call_id not in self._results:
            raise RuntimeError(f"Result for call {self._call_id} not yet available")
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


class OTCExecutor:
    """Executes OTC code with access to tools."""

    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.tool_names = set(self.tools.keys())

    def _contains_unresolved(self, value: Any) -> bool:
        """Check if a value (possibly nested) contains an unresolved PendingResult."""
        if isinstance(value, PendingResult):
            return not value.is_ready()
        if isinstance(value, list):
            return any(self._contains_unresolved(item) for item in value)
        if isinstance(value, tuple):
            return any(self._contains_unresolved(item) for item in value)
        if isinstance(value, set):
            return any(self._contains_unresolved(item) for item in value)
        if isinstance(value, dict):
            return any(self._contains_unresolved(v) for v in value.values())
        return False

    def _resolve_dependencies(self, value: Any, results: dict[int, Any]) -> Any:
        """Replace PendingResult placeholders with concrete values."""
        if isinstance(value, PendingResult):
            return value._require_result()
        if isinstance(value, list):
            return [self._resolve_dependencies(v, results) for v in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_dependencies(v, results) for v in value)
        if isinstance(value, set):
            return {self._resolve_dependencies(v, results) for v in value}
        if isinstance(value, dict):
            return {k: self._resolve_dependencies(v, results) for k, v in value.items()}
        return value

    def _resolve_output_value(self, value: Any, results: dict[int, Any]) -> Any:
        """Resolve PendingResult placeholders when building the final output."""
        if isinstance(value, PendingResult):
            return value._require_result()
        if isinstance(value, list):
            return [self._resolve_output_value(v, results) for v in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_output_value(v, results) for v in value)
        if isinstance(value, set):
            return {self._resolve_output_value(v, results) for v in value}
        if isinstance(value, dict):
            return {k: self._resolve_output_value(v, results) for k, v in value.items()}
        return value

    def _make_sync_tool_wrapper(
        self,
        tool: Tool,
        pending_calls: list,
        results: dict[int, Any],
        call_state: dict[str, int],
        pending_call_ids: set[int],
    ) -> Callable:
        """Create a sync wrapper that queues tool calls for later execution."""

        def wrapper(*args, **kwargs):
            # Convert positional args to kwargs using tool parameter order
            if args and tool.parameters:
                param_names = list(tool.parameters.keys())
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = arg

            # Ensure we don't pass unresolved PendingResult objects as arguments
            if self._contains_unresolved(kwargs):
                raise RuntimeError("Result for call dependency not yet available")

            # Resolve any PendingResult values before queueing
            resolved_kwargs = self._resolve_dependencies(kwargs, results)

            # Generate a deterministic call ID based on execution order
            call_id = call_state["next_id"]
            call_state["next_id"] += 1

            # Avoid re-queueing calls that already have results or are pending
            if call_id not in results and call_id not in pending_call_ids:
                pending_call_ids.add(call_id)
                pending_calls.append(
                    {
                        "id": call_id,
                        "tool": tool.name,
                        "kwargs": resolved_kwargs,
                    }
                )

            # Return a placeholder that will be resolved later
            return PendingResult(call_id, results)

        return wrapper

    async def _execute_pending_calls(self, pending_calls: list, results: dict) -> None:
        """Execute all pending tool calls in parallel."""
        if not pending_calls:
            return

        async def execute_one(call: dict) -> tuple[int, Any]:
            tool = self.tools[call["tool"]]
            try:
                if tool.run is None:
                    raise OTCExecutionError("tool is not executable")
                if asyncio.iscoroutinefunction(tool.run):
                    result = await tool.run(**call["kwargs"])
                else:
                    result = tool.run(**call["kwargs"])

                # Try to parse as JSON if it's a string
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        pass  # Keep as string

                return call["id"], result
            except Exception as e:
                return call["id"], {"error": str(e)}

        # Execute all in parallel
        call_results = await asyncio.gather(
            *[execute_one(call) for call in pending_calls]
        )

        # Store results
        for call_id, result in call_results:
            results[call_id] = result

        # Clear pending
        pending_calls.clear()

    async def execute(self, code: str) -> str:
        """Execute OTC code and return the final output.

        The execution model:
        1. Parse and validate the code
        2. Execute line-by-line, collecting tool calls
        3. When we hit a point where results are needed, execute pending calls
        4. Continue until done
        5. Return captured output or final expression value
        """
        # Validate
        tree = validate_code(code, self.tool_names)

        # Set up execution environment
        pending_calls: list = []
        results: dict = {}
        output_capture = OutputCapture()
        pending_call_ids: set[int] = set()
        call_state = {"next_id": 0}

        # Create tool wrappers
        tool_wrappers = {
            name: self._make_sync_tool_wrapper(
                tool, pending_calls, results, call_state, pending_call_ids
            )
            for name, tool in self.tools.items()
        }

        # Build globals
        exec_globals = {
            "__builtins__": {**SAFE_BUILTINS, "print": output_capture.print},
            "json": json,  # Allow json for output formatting
            **tool_wrappers,
        }

        exec_locals: dict = {}

        # Execute the code
        # We need to handle the deferred execution pattern:
        # Tool calls return PendingResult objects, and we need to resolve them
        # before they're actually used.

        # Strategy: Execute the whole thing, catching any "not yet available" errors,
        # then execute pending calls and retry until done.

        max_iterations = 100  # Prevent infinite loops

        for _ in range(max_iterations):
            # Reset call sequencing and pending tracking for this pass
            call_state["next_id"] = 0
            pending_call_ids.clear()
            try:
                exec(compile(tree, "<otc>", "exec"), exec_globals, exec_locals)
                # If we get here, execution completed
                # Execute any remaining pending calls (though their results won't be used)
                await self._execute_pending_calls(pending_calls, results)
                pending_call_ids.clear()
                break

            except RuntimeError as e:
                if "not yet available" in str(e):
                    # Need to resolve pending calls and retry
                    await self._execute_pending_calls(pending_calls, results)
                    pending_call_ids.clear()
                    # Continue the loop to retry
                else:
                    raise OTCExecutionError(f"Runtime error: {e}")

            except Exception as e:
                raise OTCExecutionError(f"Execution error: {type(e).__name__}: {e}")

        else:
            raise OTCExecutionError("Execution exceeded maximum iterations")

        # Get output
        output = output_capture.get_output()

        # If no print output, try to get the last expression value
        if not output and exec_locals:
            # Look for a 'result' variable or the last assigned value
            if "result" in exec_locals:
                result = self._resolve_output_value(exec_locals["result"], results)
                if isinstance(result, str):
                    output = result
                else:
                    output = json.dumps(result, default=str, indent=2)

        return output if output else "Composition completed with no output"
