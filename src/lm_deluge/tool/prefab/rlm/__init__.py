"""
RLM (Recursive Language Model) for lm-deluge.

Enables models to process long contexts through a REPL environment
with recursive LM calls, based on the RLM paper:
https://alexzhang13.github.io/blog/2025/rlm/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lm_deluge.prompt import Conversation
from lm_deluge.tool import Tool

from .executor import RLMExecutionError, RLMExecutor
from .parse import RLMSecurityError

if TYPE_CHECKING:
    from lm_deluge.api_requests.base import APIResponse
    from lm_deluge.client import _LLMClient


RLM_SYSTEM_PROMPT = """You have access to a long context stored in the variable `{context_var}`.
You can write Python code to analyze this context using the `execute` tool.

IMPORTANT RULES:
1. You MUST use print() to see output. Bare expressions produce NO output.
2. You MUST call final(answer) when you have the answer. This is required!

Available in your code environment:
- `{context_var}`: The full context as a string ({context_len:,} characters)
- `lm(prompt)`: Make a recursive LLM call (runs in parallel when possible)
- `final(answer)`: Signal completion with the given answer - YOU MUST CALL THIS!
- `final_var(varname)`: Signal completion with a variable's value
- Modules: `re`, `math`, `collections`, `json` (imports are allowed but optional)
- From collections: `Counter`, `defaultdict`, `deque`, `namedtuple`, `OrderedDict`
- Standard builtins: `len`, `str`, `int`, `list`, `dict`, `sum`, `sorted`, `map`, `filter`, etc.

Example - count word occurrences:
```python
count = len(re.findall(r'\\bword\\b', {context_var}))
print(f"Found {{count}} occurrences")
final(count)
```

Example - use Counter:
```python
words = {context_var}.split()
counts = Counter(words)
print(counts.most_common(10))
final(counts.most_common(10))
```

Example - analyze with lm() calls:
```python
chunks = [{context_var}[i:i+2000] for i in range(0, len({context_var}), 2000)][:3]
summaries = [lm(f"Summarize: {{chunk}}") for chunk in chunks]
combined = "\\n".join(str(s) for s in summaries)
final(f"Summary:\\n{{combined}}")
```

Variables persist between execute() calls. Always call final() when you have the answer!
"""


class RLMManager:
    """Manages RLM execution for a long context.

    The RLMManager exposes a REPL-like interface as tools that allow an LLM
    to analyze a long context by writing Python code.

    Example:
        >>> manager = RLMManager(
        ...     context=long_document,
        ...     client=LLMClient("gpt-4.1-mini"),  # For lm() calls
        ... )
        >>> main_client = LLMClient("gpt-4.1")
        >>> conv = Conversation.system(manager.get_system_prompt())
        >>> conv = conv.user("What are the main themes in this document?")
        >>> conv, resp = await main_client.run_agent_loop(
        ...     conv,
        ...     tools=manager.get_tools(),
        ... )
        >>> if manager.is_complete:
        ...     print(manager.final_answer)
    """

    def __init__(
        self,
        context: str,
        client: _LLMClient,
        context_var_name: str = "CONTEXT",
        max_lm_calls_per_execution: int = 20,
    ):
        """Initialize the RLMManager.

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

        self.executor = RLMExecutor(
            context=context,
            client=client,
            context_var_name=context_var_name,
            max_lm_calls_per_execution=max_lm_calls_per_execution,
        )

        self._final_answer: str | None = None
        self._tools: list[Tool] | None = None

    async def _execute(self, code: str) -> str:
        """Execute code against the context."""
        try:
            answer, is_final = await self.executor.execute(code)
            if is_final:
                self._final_answer = answer
                # Truncate for display but keep full answer stored
                display = answer[:1000] + "..." if len(answer) > 1000 else answer
                return f"[FINAL ANSWER SET]\n{display}"
            return answer
        except RLMSecurityError as e:
            return f"Security error: {e}"
        except RLMExecutionError as e:
            return f"Execution error: {e}"
        except Exception as e:
            return f"Unexpected error: {type(e).__name__}: {e}"

    def get_system_prompt(self) -> str:
        """Get the system prompt explaining the RLM environment."""
        return RLM_SYSTEM_PROMPT.format(
            context_var=self.context_var_name,
            context_len=len(self.context),
        )

    def get_tools(self) -> list[Tool]:
        """Get the tools for RLM execution."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool(
                name="execute",
                description=(
                    f"Execute Python code to analyze the context. "
                    f"The context ({len(self.context):,} chars) is available as `{self.context_var_name}`. "
                    f"Use `lm(prompt)` for recursive LLM calls (parallel when possible), and "
                    f"`final(answer)` or `final_var(varname)` to signal completion. "
                    f"Variables persist between calls. "
                    f"Modules available without import: re, math, collections, json."
                ),
                run=self._execute,
                parameters={
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                required=["code"],
            )
        ]
        return self._tools

    @property
    def is_complete(self) -> bool:
        """Check if FINAL() was called."""
        return self._final_answer is not None

    @property
    def final_answer(self) -> str | None:
        """Get the final answer if set."""
        return self._final_answer

    def reset(self) -> None:
        """Reset the RLM state."""
        self.executor.reset()
        self._final_answer = None


@dataclass
class RLMResult:
    """Result from RLMPipeline."""

    answer: str
    conversation: Conversation
    rounds_used: int
    final_response: APIResponse


class RLMPipeline:
    """High-level pipeline for RLM processing.

    A thin wrapper that takes a long context and question, sets up an RLMManager,
    runs an agent loop until final() is called, and returns the result.

    Example:
        >>> pipeline = RLMPipeline(
        ...     context=long_document,
        ...     client=LLMClient("gpt-4.1"),  # Smart orchestrator
        ...     lm_client=LLMClient("gpt-4.1-mini"),  # Cheaper model for lm() calls
        ...     question="What are the main themes in this document?",
        ... )
        >>> result = await pipeline.run()
        >>> print(result.answer)
    """

    def __init__(
        self,
        context: str,
        client: _LLMClient,
        question: str,
        *,
        lm_client: _LLMClient | None = None,
        context_var_name: str = "CONTEXT",
        max_rounds: int = 15,
        max_lm_calls_per_execution: int = 20,
    ):
        """Initialize the RLMPipeline.

        Args:
            context: The long context string to analyze
            client: LLMClient for the main agent (runs the execute loop)
            question: The question to answer about the context
            lm_client: LLMClient for lm() calls (defaults to same as client)
            context_var_name: Variable name for the context (default: "CONTEXT")
            max_rounds: Maximum agent loop rounds (default: 15)
            max_lm_calls_per_execution: Maximum lm() calls per execute() call
        """
        self.context = context
        self.client = client
        self.lm_client = lm_client or client
        self.question = question
        self.context_var_name = context_var_name
        self.max_rounds = max_rounds
        self.max_lm_calls_per_execution = max_lm_calls_per_execution

    async def run(self) -> RLMResult:
        """Run the RLM pipeline until completion."""
        manager = RLMManager(
            context=self.context,
            client=self.lm_client,
            context_var_name=self.context_var_name,
            max_lm_calls_per_execution=self.max_lm_calls_per_execution,
        )

        # Build conversation with system prompt and question
        conv = Conversation.system(manager.get_system_prompt())
        conv = conv.user(
            f"Question to answer about the context:\n\n{self.question}\n\n"
            "Use the execute tool to analyze the context and find the answer. "
            "Start by peeking at the context structure, then use appropriate "
            "techniques (regex, chunking, lm() calls) to find the answer. "
            "Call final(answer) when you have the answer."
        )

        # Run agent loop
        conv, resp = await self.client.run_agent_loop(
            conv,
            tools=manager.get_tools(),
            max_rounds=self.max_rounds,
        )

        # Extract answer
        if manager.is_complete:
            answer = manager.final_answer or "No answer produced"
        else:
            # Model stopped without calling final() - use last response
            answer = resp.completion or "No answer produced (final not called)"

        # Count rounds used
        rounds_used = sum(1 for m in conv.messages if m.role == "assistant")

        return RLMResult(
            answer=answer,
            conversation=conv,
            rounds_used=rounds_used,
            final_response=resp,
        )


__all__ = [
    "RLMManager",
    "RLMPipeline",
    "RLMResult",
    "RLMExecutor",
    "RLMExecutionError",
    "RLMSecurityError",
]
