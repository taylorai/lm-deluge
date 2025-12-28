"""Tests for MCPServer.from_mcp_config"""

from lm_deluge.tool import MCPServer


def test_from_mcp_config_basic():
    """Test creating MCPServer instances from a config dict."""
    config = {
        "mcpServers": {
            "weather": {"url": "https://weather.example.com/mcp"},
            "search": {
                "url": "https://search.example.com/mcp",
                "token": "secret-token",
            },
        }
    }

    servers = MCPServer.from_mcp_config(config)

    assert len(servers) == 2

    # Find servers by name
    weather = next(s for s in servers if s.name == "weather")
    search = next(s for s in servers if s.name == "search")

    assert weather.url == "https://weather.example.com/mcp"
    assert weather.token is None

    assert search.url == "https://search.example.com/mcp"
    assert search.token == "secret-token"


def test_from_mcp_config_without_wrapper():
    """Test passing just the mcpServers block directly."""
    config = {
        "weather": {"url": "https://weather.example.com/mcp"},
    }

    servers = MCPServer.from_mcp_config(config)

    assert len(servers) == 1
    assert servers[0].name == "weather"
    assert servers[0].url == "https://weather.example.com/mcp"


def test_from_mcp_config_skips_command_servers():
    """Test that command-based servers are skipped."""
    config = {
        "mcpServers": {
            "url_server": {"url": "https://example.com/mcp"},
            "command_server": {"command": "python", "args": ["./server.py"]},
        }
    }

    servers = MCPServer.from_mcp_config(config)

    assert len(servers) == 1
    assert servers[0].name == "url_server"


def test_from_mcp_config_with_all_options():
    """Test with all possible options set."""
    config = {
        "mcpServers": {
            "full": {
                "url": "https://example.com/mcp",
                "token": "auth-token",
                "configuration": {"enabled_tools": ["tool1"]},
                "headers": {"X-Custom": "value"},
            }
        }
    }

    servers = MCPServer.from_mcp_config(config)

    assert len(servers) == 1
    server = servers[0]
    assert server.name == "full"
    assert server.url == "https://example.com/mcp"
    assert server.token == "auth-token"
    assert server.configuration == {"enabled_tools": ["tool1"]}
    assert server.headers == {"X-Custom": "value"}


def test_from_mcp_config_empty():
    """Test with empty config."""
    servers = MCPServer.from_mcp_config({})
    assert servers == []

    servers = MCPServer.from_mcp_config({"mcpServers": {}})
    assert servers == []


if __name__ == "__main__":
    test_from_mcp_config_basic()
    test_from_mcp_config_without_wrapper()
    test_from_mcp_config_skips_command_servers()
    test_from_mcp_config_with_all_options()
    test_from_mcp_config_empty()
    print("All tests passed!")
