from lm_deluge.api_requests.base import APIResponse
from lm_deluge.client import AgentLoopResponse, _LLMClient
from lm_deluge.prompt import Conversation, prompts_to_conversations
from lm_deluge.tool import Tool


class SubAgentManager:
    """Manages subagent tasks that can be spawned by a main LLM via tool calls.

    The SubAgentManager exposes tools that allow a main LLM to delegate subtasks
    to specialized or cheaper subagent models, saving context and improving efficiency.

    Example:
        >>> manager = SubAgentManager(
        ...     client=LLMClient("gpt-4o-mini"),  # Subagent model
        ...     tools=[search_tool, calculator_tool]  # Tools available to subagents
        ... )
        >>> main_client = LLMClient("gpt-4o")  # More expensive main model
        >>> conv = Conversation.user("Research AI and calculate market size")
        >>> # Main model can now call manager tools to spawn subagents
        >>> conv, resp = await main_client.run_agent_loop(
        ...     conv,
        ...     tools=manager.get_tools()
        ... )
    """

    def __init__(
        self,
        client: _LLMClient,
        tools: list[Tool] | None = None,
        max_rounds: int = 5,
    ):
        """Initialize the SubAgentManager.

        Args:
            client: LLMClient to use for subagent tasks
            tools: Tools available to subagents (optional)
            max_rounds: Maximum rounds for each subagent's agent loop
        """
        self.client = client
        self.tools = tools or []
        self.max_rounds = max_rounds
        self.subagents: dict[int, dict] = {}

    async def _start_subagent(self, task: str) -> int:
        """Start a subagent with the given task.

        Args:
            task: The task description for the subagent

        Returns:
            Subagent task ID
        """
        conversation = prompts_to_conversations([task])[0]
        assert isinstance(conversation, Conversation)

        # Use agent loop nowait API to start the subagent
        task_id = self.client.start_agent_loop_nowait(
            conversation,
            tools=self.tools,  # type: ignore
            max_rounds=self.max_rounds,
        )

        # Track the subagent
        self.subagents[task_id] = {
            "status": "running",
            "conversation": None,
            "response": None,
            "error": None,
        }

        return task_id

    def _finalize_subagent_result(
        self, agent_id: int, result: AgentLoopResponse
    ) -> str:
        """Update subagent tracking state from a finished agent loop."""
        agent = self.subagents[agent_id]
        agent["conversation"] = result.conversation
        agent["response"] = result.final_response

        if result.final_response.is_error:
            agent["status"] = "error"
            agent["error"] = result.final_response.error_message
            return f"Error: {agent['error']}"

        agent["status"] = "finished"
        return result.final_response.completion or "Subagent finished with no output"

    async def _check_subagent(self, agent_id: int) -> str:
        """Check the status of a subagent.

        Args:
            agent_id: The subagent task ID

        Returns:
            Status string describing the subagent's state
        """
        if agent_id not in self.subagents:
            return f"Error: Subagent {agent_id} not found"

        agent = self.subagents[agent_id]
        status = agent["status"]

        if status == "finished":
            response: APIResponse = agent["response"]
            return response.completion or "Subagent finished with no output"
        elif status == "error":
            return f"Error: {agent['error']}"
        else:
            # Try to check if it's done
            try:
                # Check if the task exists in client's results
                stored_result = self.client._results.get(agent_id)
                if isinstance(stored_result, AgentLoopResponse):
                    return self._finalize_subagent_result(agent_id, stored_result)

                task = self.client._tasks.get(agent_id)
                if task and task.done():
                    try:
                        task_result = task.result()
                    except Exception as e:
                        agent["status"] = "error"
                        agent["error"] = str(e)
                        return f"Error: {agent['error']}"

                    if isinstance(task_result, AgentLoopResponse):
                        return self._finalize_subagent_result(agent_id, task_result)

                    agent["status"] = "error"
                    agent["error"] = (
                        f"Unexpected task result type: {type(task_result).__name__}"
                    )
                    return f"Error: {agent['error']}"

                # Still running
                return f"Subagent {agent_id} is still running. Call this tool again to check status."
            except Exception as e:
                agent["status"] = "error"
                agent["error"] = str(e)
                return f"Error checking subagent: {e}"

    async def _wait_for_subagent(self, agent_id: int) -> str:
        """Wait for a subagent to complete and return its output.

        Args:
            agent_id: The subagent task ID

        Returns:
            The subagent's final output
        """
        if agent_id not in self.subagents:
            return f"Error: Subagent {agent_id} not found"

        try:
            # Use the wait_for_agent_loop API
            conversation, response = await self.client.wait_for_agent_loop(agent_id)

            agent = self.subagents[agent_id]
            agent["conversation"] = conversation
            agent["response"] = response

            if response.is_error:
                agent["status"] = "error"
                agent["error"] = response.error_message
                return f"Error: {response.error_message}"
            else:
                agent["status"] = "finished"
                return response.completion or "Subagent finished with no output"
        except Exception as e:
            agent = self.subagents[agent_id]
            agent["status"] = "error"
            agent["error"] = str(e)
            return f"Error waiting for subagent: {e}"

    def get_tools(self) -> list[Tool]:
        """Get the tools that allow a main LLM to control subagents.

        Returns:
            List of Tool objects for starting, checking, and waiting for subagents
        """
        start_tool = Tool(
            name="start_subagent",
            description=(
                "Start a subagent to work on a subtask independently. "
                "Use this to delegate complex subtasks or when you need to save context. "
                "Returns the subagent's task ID which can be used to check its status."
            ),
            run=self._start_subagent,
            parameters={
                "task": {
                    "type": "string",
                    "description": "The task description for the subagent to work on",
                }
            },
            required=["task"],
        )

        check_tool = Tool(
            name="check_subagent",
            description=(
                "Check the status and output of a running subagent. "
                "If the subagent is still running, you'll be told to check again later. "
                "If finished, returns the subagent's final output."
            ),
            run=self._check_subagent,
            parameters={
                "agent_id": {
                    "type": "integer",
                    "description": "The task ID of the subagent to check",
                }
            },
            required=["agent_id"],
        )

        wait_tool = Tool(
            name="wait_for_subagent",
            description=(
                "Wait for a subagent to complete and return its output. "
                "This will block until the subagent finishes. "
                "Use check_subagent if you want to do other work while waiting."
            ),
            run=self._wait_for_subagent,
            parameters={
                "agent_id": {
                    "type": "integer",
                    "description": "The task ID of the subagent to wait for",
                }
            },
            required=["agent_id"],
        )

        return [start_tool, check_tool, wait_tool]
