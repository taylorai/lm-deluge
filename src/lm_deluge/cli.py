"""
LM-Deluge CLI

Usage:
    deluge list [--provider PROVIDER] [--name NAME] [--auto] [--json] ...
    deluge run MODEL [--input INPUT | --file FILE] [--max-tokens N] [--temperature T] ...
    deluge agent MODEL [--mcp-config FILE] [--prefab TOOLS] [--input INPUT] ...

Examples:
    deluge list
    deluge list --auto                            # Only models with API keys set
    deluge list --provider anthropic --reasoning
    deluge list --name claude --json
    deluge run claude-3.5-haiku -i "What is 2+2?"
    echo "Hello" | deluge run gpt-4.1-mini
    deluge run claude-4-sonnet --file prompt.txt --max-tokens 4096
    deluge agent claude-3.5-haiku --mcp-config mcp.json -i "Search for AI news"
    deluge agent claude-4-sonnet --prefab todo,memory -i "Create a task list"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .models import find_models, APIModel
from .client import LLMClient
from .prompt import Conversation


def _model_to_dict(model: APIModel) -> dict[str, Any]:
    """Convert APIModel to a JSON-serializable dict."""
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "api_spec": model.api_spec,
        "input_cost": model.input_cost,
        "output_cost": model.output_cost,
        "supports_json": model.supports_json,
        "supports_images": model.supports_images,
        "supports_logprobs": model.supports_logprobs,
        "reasoning_model": model.reasoning_model,
    }


def cmd_list(args: argparse.Namespace) -> int:
    """List models matching the given criteria."""
    # Convert boolean flags: only pass True if set, None otherwise
    models = find_models(
        provider=args.provider,
        supports_json=True if args.json_mode else None,
        supports_images=True if args.images else None,
        supports_logprobs=True if args.logprobs else None,
        reasoning_model=True if args.reasoning else None,
        min_input_cost=args.min_input_cost,
        max_input_cost=args.max_input_cost,
        min_output_cost=args.min_output_cost,
        max_output_cost=args.max_output_cost,
        name_contains=args.name,
        has_api_key=True if args.auto else None,
        sort_by=args.sort,
        limit=args.limit,
    )

    if args.json:
        output = [_model_to_dict(m) for m in models]
        print(json.dumps(output, indent=2))
    else:
        if not models:
            print("No models found matching criteria.", file=sys.stderr)
            return 0

        # Calculate column widths
        id_width = max(len(m.id) for m in models)
        provider_width = max(len(m.provider) for m in models)

        # Header
        print(
            f"{'MODEL':<{id_width}}  {'PROVIDER':<{provider_width}}  {'INPUT $/M':>10}  {'OUTPUT $/M':>10}  FLAGS"
        )
        print("-" * (id_width + provider_width + 40))

        for m in models:
            flags = []
            if m.supports_json:
                flags.append("json")
            if m.supports_images:
                flags.append("img")
            if m.supports_logprobs:
                flags.append("logp")
            if m.reasoning_model:
                flags.append("reason")

            input_cost = f"${m.input_cost:.2f}" if m.input_cost is not None else "N/A"
            output_cost = (
                f"${m.output_cost:.2f}" if m.output_cost is not None else "N/A"
            )

            print(
                f"{m.id:<{id_width}}  {m.provider:<{provider_width}}  {input_cost:>10}  {output_cost:>10}  {','.join(flags)}"
            )

        print(f"\nTotal: {len(models)} models")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a model on input and output JSON to stdout."""
    # Determine input text
    if args.input:
        prompt_text = args.input
    elif args.file:
        try:
            with open(args.file, "r") as f:
                prompt_text = f.read()
        except FileNotFoundError:
            print(
                json.dumps({"error": f"File not found: {args.file}"}), file=sys.stdout
            )
            return 1
        except Exception as e:
            print(json.dumps({"error": f"Failed to read file: {e}"}), file=sys.stdout)
            return 1
    elif not sys.stdin.isatty():
        prompt_text = sys.stdin.read()
    else:
        print(
            json.dumps(
                {"error": "No input provided. Use --input, --file, or pipe to stdin."}
            ),
            file=sys.stdout,
        )
        return 1

    if not prompt_text.strip():
        print(json.dumps({"error": "Empty input provided."}), file=sys.stdout)
        return 1

    # Build conversation
    image = args.image if hasattr(args, "image") else None
    if args.system:
        conv = Conversation().system(args.system).user(prompt_text, image=image)
    else:
        conv = Conversation().user(prompt_text, image=image)

    # Build client params
    client_kwargs: dict[str, Any] = {
        "model_names": args.model,
        "max_new_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        client_kwargs["temperature"] = args.temperature

    try:
        client = LLMClient(**client_kwargs)
        client.open(show_progress=False)
        response = asyncio.run(client.start(conv))
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        return 1
    except Exception as e:
        print(json.dumps({"error": f"Request failed: {e}"}), file=sys.stdout)
        return 1

    # Build output
    output: dict[str, Any] = {
        "model": args.model,
        "completion": response.completion if response.completion else None,
        "is_error": response.is_error,
    }

    if response.is_error:
        output["error_message"] = response.error_message

    if response.usage:
        output["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    if response.cost is not None:
        output["cost"] = response.cost

    if args.verbose and response.finish_reason:
        output["finish_reason"] = response.finish_reason

    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0 if not response.is_error else 1


def _print_json(obj: dict[str, Any]) -> None:
    """Print JSON and flush immediately for streaming."""
    print(json.dumps(obj), flush=True)


def cmd_agent(args: argparse.Namespace) -> int:
    """Run an agent loop with tools and output JSON blocks for each content piece."""
    from .tool import Tool, MCPServer
    from .prompt.text import Text
    from .prompt.tool_calls import ToolCall
    from .prompt.thinking import Thinking

    # Determine input text
    if args.input:
        prompt_text = args.input
    elif args.file:
        try:
            with open(args.file, "r") as f:
                prompt_text = f.read()
        except FileNotFoundError:
            _print_json({"type": "error", "error": f"File not found: {args.file}"})
            return 1
        except Exception as e:
            _print_json({"type": "error", "error": f"Failed to read file: {e}"})
            return 1
    elif not sys.stdin.isatty():
        prompt_text = sys.stdin.read()
    else:
        _print_json(
            {
                "type": "error",
                "error": "No input provided. Use --input, --file, or pipe to stdin.",
            }
        )
        return 1

    if not prompt_text.strip():
        _print_json({"type": "error", "error": "Empty input provided."})
        return 1

    def print_message_parts(msg_role: str, parts: list) -> None:
        """Print JSON for each part of a message."""
        for part in parts:
            if isinstance(part, Text):
                _print_json({"type": "text", "role": msg_role, "content": part.text})
            elif isinstance(part, ToolCall):
                _print_json(
                    {
                        "type": "tool_call",
                        "id": part.id,
                        "name": part.name,
                        "arguments": part.arguments,
                    }
                )
            elif isinstance(part, Thinking):
                _print_json({"type": "thinking", "content": part.content})

    async def run_agent() -> int:
        tools: list[Any] = []
        tool_map: dict[str, Tool] = {}

        # Load MCP tools from config
        if args.mcp_config:
            try:
                import json5

                with open(args.mcp_config, "r") as f:
                    mcp_config = json5.load(f)
                # URL-based servers -> MCPServer objects (provider-native)
                mcp_servers = MCPServer.from_mcp_config(mcp_config)
                tools.extend(mcp_servers)
                # Expand MCP servers to tools for local execution
                for server in mcp_servers:
                    server_tools = await server.to_tools()
                    for t in server_tools:
                        tool_map[t.name] = t
                # Command-based servers -> Tool objects (local execution)
                cmd_tools = await Tool.from_mcp_config(mcp_config)
                tools.extend(cmd_tools)
                for t in cmd_tools:
                    tool_map[t.name] = t
            except FileNotFoundError:
                _print_json(
                    {
                        "type": "error",
                        "error": f"MCP config not found: {args.mcp_config}",
                    }
                )
                return 1
            except Exception as e:
                _print_json(
                    {"type": "error", "error": f"Failed to load MCP config: {e}"}
                )
                return 1

        # Load prefab tools
        if args.prefab:
            prefab_names = [p.strip() for p in args.prefab.split(",")]
            for name in prefab_names:
                try:
                    prefab_tools: list[Tool] = []
                    if name == "todo":
                        from .tool.prefab import TodoManager

                        prefab_tools = TodoManager().get_tools()
                    elif name == "memory":
                        from .tool.prefab.memory import MemoryManager

                        prefab_tools = MemoryManager().get_tools()
                    elif name == "filesystem":
                        from .tool.prefab import FilesystemManager

                        prefab_tools = FilesystemManager().get_tools()
                    elif name == "sandbox":
                        import platform

                        if platform.system() == "Darwin":
                            from .tool.prefab.sandbox import SeatbeltSandbox

                            sandbox = SeatbeltSandbox()
                            await sandbox.__aenter__()
                            prefab_tools = sandbox.get_tools()
                        else:
                            from .tool.prefab.sandbox import DockerSandbox

                            sandbox = DockerSandbox()
                            await sandbox.__aenter__()
                            prefab_tools = sandbox.get_tools()
                    else:
                        _print_json(
                            {
                                "type": "error",
                                "error": f"Unknown prefab tool: {name}. Available: todo, memory, filesystem, sandbox",
                            }
                        )
                        return 1
                    tools.extend(prefab_tools)
                    for t in prefab_tools:
                        tool_map[t.name] = t
                except ImportError as e:
                    _print_json(
                        {
                            "type": "error",
                            "error": f"Failed to load prefab '{name}': {e}",
                        }
                    )
                    return 1

        # Build conversation
        image = args.image if hasattr(args, "image") else None
        if args.system:
            conv = Conversation().system(args.system).user(prompt_text, image=image)
        else:
            conv = Conversation().user(prompt_text, image=image)

        # Print initial user message
        _print_json({"type": "text", "role": "user", "content": prompt_text})

        # Build client
        client_kwargs: dict[str, Any] = {
            "model_names": args.model,
            "max_new_tokens": args.max_tokens,
        }
        if args.temperature is not None:
            client_kwargs["temperature"] = args.temperature

        try:
            client = LLMClient(**client_kwargs)
            client.open(show_progress=False)

            # Manual agent loop with streaming output
            total_usage = {"input_tokens": 0, "output_tokens": 0}
            total_cost = 0.0
            last_response = None
            round_num = 0

            for round_num in range(args.max_rounds):
                # Get model response
                response = await client.start(conv, tools=tools)
                last_response = response

                if response.is_error:
                    _print_json({"type": "error", "error": response.error_message})
                    break

                # Track usage
                if response.usage:
                    total_usage["input_tokens"] += response.usage.input_tokens or 0
                    total_usage["output_tokens"] += response.usage.output_tokens or 0
                if response.cost:
                    total_cost += response.cost

                # Print assistant response parts
                if response.content:
                    print_message_parts("assistant", response.content.parts)

                    # Check for tool calls
                    tool_calls = response.content.tool_calls
                    if not tool_calls:
                        # No tool calls, we're done
                        break

                    # Add assistant message to conversation
                    conv = conv.add(response.content)

                    # Execute tool calls and print results
                    for call in tool_calls:
                        tool_obj = tool_map.get(call.name)
                        if tool_obj:
                            try:
                                result = await tool_obj.acall(**call.arguments)
                                result_str = (
                                    result
                                    if isinstance(result, str)
                                    else json.dumps(result)
                                )
                            except Exception as e:
                                result_str = f"Error: {e}"
                        else:
                            result_str = f"Error: Unknown tool '{call.name}'"

                        _print_json(
                            {
                                "type": "tool_result",
                                "tool_call_id": call.id,
                                "name": call.name,
                                "result": result_str,
                            }
                        )

                        # Add tool result to conversation
                        conv = conv.with_tool_result(call.id, result_str)
                else:
                    # No content, we're done
                    break

            # Final summary
            done_output: dict[str, Any] = {"type": "done", "rounds": round_num + 1}
            if total_usage["input_tokens"] or total_usage["output_tokens"]:
                done_output["usage"] = total_usage
            if total_cost > 0:
                done_output["cost"] = total_cost
            if last_response and last_response.is_error:
                done_output["error"] = last_response.error_message
            _print_json(done_output)

            return 0 if (last_response and not last_response.is_error) else 1

        except ValueError as e:
            _print_json({"type": "error", "error": str(e)})
            return 1
        except Exception as e:
            _print_json({"type": "error", "error": f"Agent loop failed: {e}"})
            return 1

    return asyncio.run(run_agent())


def main():
    parser = argparse.ArgumentParser(
        prog="deluge",
        description="LM-Deluge CLI - Run and manage LLM models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- list command ----
    list_parser = subparsers.add_parser(
        "list",
        help="List available models",
        description="List and filter available models in the registry",
    )
    list_parser.add_argument(
        "--provider",
        type=str,
        help="Filter by provider/api_spec (e.g., openai, anthropic, google)",
    )
    list_parser.add_argument(
        "--name",
        type=str,
        help="Filter by substring in model ID (case-insensitive)",
    )
    list_parser.add_argument(
        "--json-mode",
        action="store_true",
        dest="json_mode",
        help="Only show models that support JSON mode",
    )
    list_parser.add_argument(
        "--images",
        action="store_true",
        help="Only show models that support image inputs",
    )
    list_parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Only show models that support logprobs",
    )
    list_parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Only show reasoning models",
    )
    list_parser.add_argument(
        "--auto",
        action="store_true",
        help="Only show models whose provider API key is set in the environment",
    )
    list_parser.add_argument(
        "--min-input-cost",
        type=float,
        help="Minimum input cost ($ per million tokens)",
    )
    list_parser.add_argument(
        "--max-input-cost",
        type=float,
        help="Maximum input cost ($ per million tokens)",
    )
    list_parser.add_argument(
        "--min-output-cost",
        type=float,
        help="Minimum output cost ($ per million tokens)",
    )
    list_parser.add_argument(
        "--max-output-cost",
        type=float,
        help="Maximum output cost ($ per million tokens)",
    )
    list_parser.add_argument(
        "--sort",
        type=str,
        choices=["input_cost", "output_cost", "-input_cost", "-output_cost"],
        help="Sort by cost (prefix with - for descending)",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of results",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    list_parser.set_defaults(func=cmd_list)

    # ---- run command ----
    run_parser = subparsers.add_parser(
        "run",
        help="Run a model on input",
        description="Run a model on input and output JSON to stdout",
    )
    run_parser.add_argument(
        "model",
        type=str,
        help="Model ID to use (e.g., claude-3.5-haiku, gpt-4.1-mini)",
    )
    input_group = run_parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input text (inline)",
    )
    input_group.add_argument(
        "--file",
        "-f",
        type=str,
        help="Read input from file",
    )
    run_parser.add_argument(
        "--system",
        "-s",
        type=str,
        help="System prompt",
    )
    run_parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to include with the prompt",
    )
    run_parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    run_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Sampling temperature",
    )
    run_parser.add_argument(
        "--pretty",
        "-p",
        action="store_true",
        help="Pretty-print JSON output",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include additional response metadata",
    )
    run_parser.set_defaults(func=cmd_run)

    # ---- agent command ----
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run an agent loop with tools",
        description="Run an agent loop with MCP servers and/or prefab tools",
    )
    agent_parser.add_argument(
        "model",
        type=str,
        help="Model ID to use (e.g., claude-3.5-haiku, gpt-4.1-mini)",
    )
    agent_input_group = agent_parser.add_mutually_exclusive_group()
    agent_input_group.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input text (inline)",
    )
    agent_input_group.add_argument(
        "--file",
        "-f",
        type=str,
        help="Read input from file",
    )
    agent_parser.add_argument(
        "--system",
        "-s",
        type=str,
        help="System prompt",
    )
    agent_parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to include with the prompt",
    )
    agent_parser.add_argument(
        "--mcp-config",
        type=str,
        help="Path to MCP config file (Claude Desktop format JSON)",
    )
    agent_parser.add_argument(
        "--prefab",
        type=str,
        help="Comma-separated prefab tools: todo,memory,filesystem,sandbox",
    )
    agent_parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum agent loop iterations (default: 10)",
    )
    agent_parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=4096,
        help="Maximum tokens to generate per response (default: 4096)",
    )
    agent_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Sampling temperature",
    )
    agent_parser.set_defaults(func=cmd_agent)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
