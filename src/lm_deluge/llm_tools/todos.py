# Adapted from https://github.com/sst/opencode - MIT License
# MIT License
# Copyright (c) 2025 opencode

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, field_validator

from ..tool import Tool

TODO_WRITE_DESCRIPTION = """Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos. Feel free to edit the todo list based on new information.
6. After completing a task - Mark it complete and add any new follow-up tasks
7. When you start working on a new task, mark the todo as in_progress. Ideally you should only have one todo as in_progress at a time. Complete existing tasks before starting new ones.

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Examples of When to Use the Todo List

<example>
User: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!
Assistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.
*Creates todo list with the following items:*
1. Create dark mode toggle component in Settings page
2. Add dark mode state management (context/store)
3. Implement CSS-in-JS styles for dark theme
4. Update existing components to support theme switching
5. Run tests and build process, addressing any failures or errors that occur
*Begins working on the first task*

<reasoning>
The assistant used the todo list because:
1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes
2. The user explicitly requested tests and build be run afterward
3. The assistant inferred that tests and build need to pass by adding "Ensure tests and build succeed" as the final task
</reasoning>
</example>

<example>
User: Help me rename the function getCwd to getCurrentWorkingDirectory across my project
Assistant: Let me first search through your codebase to find all occurrences of 'getCwd'.
*Uses grep or search tools to locate all instances of getCwd in the codebase*
Assistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.
*Creates todo list with specific items for each file that needs updating*

<reasoning>
The assistant used the todo list because:
1. First, the assistant searched to understand the scope of the task
2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps
3. The todo list helps ensure every instance is tracked and updated systematically
4. This approach prevents missing any occurrences and maintains code consistency
</reasoning>
</example>


<example>
User: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.
Assistant: I'll help implement these features. First, let's add all the features to the todo list.
*Creates a todo list breaking down each feature into specific tasks based on the project architecture*
Assistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.

<reasoning>
The assistant used the todo list because:
1. The user provided multiple complex features to implement in a comma separated list
2. The todo list helps organize these large features into manageable tasks
3. This approach allows for tracking progress across the entire implementation
</reasoning>
</example>

<example>
User: Can you help optimize my React application? It's rendering slowly and has performance issues.</user>
Assistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.
*Reviews component structure, render patterns, state management, and data fetching*
Assistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.
*Creates todo list with items like: 1) Implement memoization for expensive calculations in ProductList, 2) Add virtualization for long lists in Dashboard, 3) Optimize image loading in Gallery component, 4) Fix state update loops in ShoppingCart, 5) Review bundle size and implement code splitting*
Let's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>

<reasoning>
The assistant used the todo list because:
1. First, the assistant examined the codebase to identify specific performance issues
2. Based on this analysis, it identified multiple optimization opportunities
3. Performance optimization is a non-trivial task requiring multiple steps
4. The todo list helps methodically track improvements across different components
5. This systematic approach ensures all performance bottlenecks are addressed
</reasoning>
</example>

## Examples of When NOT to Use the Todo List

<example>
User: How do I print 'Hello World' in Python?
Assistant: In Python, you can print "Hello World" with this simple code:

python
print("Hello World")

This will output the text "Hello World" to the console when executed.</assistant>

<reasoning>
The assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.
</reasoning>
</example>

<example>
User: What does the git status command do?
Assistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.

<reasoning>
The assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.
</reasoning>
</example>

<example>
User: Can you add a comment to the calculateTotal function to explain what it does?
Assistant: Sure, let me add a comment to the calculateTotal function to explain what it does.
* Uses the Edit tool to add a comment to the calculateTotal function *

<reasoning>
The assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.
</reasoning>
</example>

<example>
User: Run npm install for me and tell me what happens.
Assistant: I'll run the npm install command for you.

*Executes: npm install*

The command completed successfully. Here's the output:
[Output of npm install command]

All dependencies have been installed according to your package.json file.

<reasoning>
The assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.
</reasoning>
</example>

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully
   - cancelled: Task no longer needed

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Only have ONE task in_progress at any time
   - Complete current tasks before starting new ones
   - Cancel tasks that become irrelevant

3. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
"""

TODO_READ_DESCRIPTION = "Use this tool to read your todo list"

TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
TodoPriority = Literal["high", "medium", "low"]


class TodoItem(BaseModel):
    """Structured representation of a single todo entry."""

    content: str = Field(description="Brief description of the task")
    status: TodoStatus = Field(
        default="pending",
        description="Current status of the task: pending, in_progress, completed, cancelled",
    )
    priority: TodoPriority = Field(
        default="medium",
        description="Priority level of the task: high, medium, low",
    )
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the todo item",
    )

    @field_validator("status", "priority", mode="before")
    @classmethod
    def _normalize_lower(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def is_active(self) -> bool:
        return self.status not in {"completed", "cancelled"}


TodoLike = TodoItem | dict[str, Any]


class TodoManager:
    """Stateful todo scratchpad that exposes read/write tools."""

    def __init__(
        self,
        todos: Sequence[TodoLike] | None = None,
        *,
        write_tool_name: str = "todowrite",
        read_tool_name: str = "todoread",
    ):
        self.write_tool_name = write_tool_name
        self.read_tool_name = read_tool_name
        self._todos: list[TodoItem] = []
        self._tools: list[Tool] | None = None

        if todos:
            self._todos = [self._coerce(todo) for todo in todos]

    def _coerce(self, todo: TodoLike) -> TodoItem:
        if isinstance(todo, TodoItem):
            return todo
        if isinstance(todo, dict):
            return TodoItem(**todo)
        raise TypeError("Todos must be TodoItem instances or dictionaries")

    def _serialize(self) -> list[dict[str, Any]]:
        return [todo.model_dump() for todo in self._todos]

    def _pending_count(self) -> int:
        return sum(1 for todo in self._todos if todo.is_active())

    def _format_output(self) -> str:
        payload = {
            "title": f"{self._pending_count()} todos",
            "todos": self._serialize(),
        }
        return json.dumps(payload, indent=2)

    def _write_tool(self, todos: list[dict[str, Any]]) -> str:
        self._todos = [self._coerce(todo) for todo in todos]
        return self._format_output()

    def _read_tool(self) -> str:
        return self._format_output()

    def get_todos(self) -> list[TodoItem]:
        """Return a copy of the current todo list."""
        return list(self._todos)

    def get_tools(self) -> list[Tool]:
        """Return Tool instances bound to this manager's state."""
        if self._tools is not None:
            return self._tools

        todo_definition = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Brief description of the task",
                },
                "status": {
                    "type": "string",
                    "description": "Current status of the task: pending, in_progress, completed, cancelled",
                    "enum": ["pending", "in_progress", "completed", "cancelled"],
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level of the task: high, medium, low",
                    "enum": ["high", "medium", "low"],
                },
                "id": {
                    "type": "string",
                    "description": "Unique identifier for the todo item",
                },
            },
            "required": ["content", "status", "priority", "id"],
        }

        write_tool = Tool(
            name=self.write_tool_name,
            description=TODO_WRITE_DESCRIPTION,
            parameters={
                "todos": {
                    "type": "array",
                    "description": "The updated todo list",
                    "items": {"$ref": "#/$defs/Todo"},
                }
            },
            required=["todos"],
            definitions={"Todo": todo_definition},
            run=self._write_tool,
        )

        read_tool = Tool(
            name=self.read_tool_name,
            description=TODO_READ_DESCRIPTION,
            parameters={},
            run=self._read_tool,
        )

        self._tools = [write_tool, read_tool]
        return self._tools


__all__ = ["TodoManager", "TodoItem", "TodoStatus", "TodoPriority"]
