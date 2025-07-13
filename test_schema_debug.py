import asyncio
import json
from lm_deluge.tool import MCPServer


async def main():
    server = MCPServer.from_openai(
        {
            "type": "mcp",
            "server_label": "pdf_plan_index",
            "server_url": "https://taylorai--pdf-index-mcp-start-server.modal.run/mcp/",
            "require_approval": "never",
            "headers": None,
        }
    )

    try:
        tools = await server.to_tools()
        for tool in tools:
            if "read_pdfs" in tool.name:
                print(f"Tool name: {tool.name}")
                print(f"Raw parameters: {json.dumps(tool.parameters, indent=2)}")
                print(f"Tool required: {tool.required}")

                # Test the openai-completions dump
                openai_dump = tool.dump_for("openai-completions")
                print("\nOpenAI completions dump (strict=True):")
                print(json.dumps(openai_dump, indent=2))

                # Test non-strict mode
                openai_dump_non_strict = tool.dump_for(
                    "openai-completions", strict=False
                )
                print("\nOpenAI completions dump (strict=False):")
                print(json.dumps(openai_dump_non_strict, indent=2))

                # Check if required keys match properties
                properties_keys = set(
                    openai_dump["function"]["parameters"]["properties"].keys()
                )
                required_keys = set(openai_dump["function"]["parameters"]["required"])
                print(f"\nProperties keys: {properties_keys}")
                print(f"Required keys: {required_keys}")
                print(f"Extra required keys: {required_keys - properties_keys}")
                print(f"Missing required keys: {properties_keys - required_keys}")

                # Test the schema generation
                schema = tool._json_schema(include_additional_properties=True)
                print("\nGenerated schema:")
                print(json.dumps(schema, indent=2))
                break
        else:
            print("No read_pdfs tool found")
            print("Available tools:")
            for tool in tools:
                print(f"  - {tool.name}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
