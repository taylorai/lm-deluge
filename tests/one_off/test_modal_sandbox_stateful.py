"""Tests for ModalSandbox stateful mode using PTY."""

import asyncio

import dotenv

from lm_deluge.tool.prefab.sandbox import ModalSandbox

dotenv.load_dotenv()


async def test_stateless_mode_no_state_persistence():
    """Verify that stateless mode (default) does NOT persist state between commands."""
    print("\n=== Testing stateless mode (no state persistence) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=False)

    # Set a variable
    await sandbox._exec(command="export MY_VAR=hello123")

    # Try to read it - should NOT work in stateless mode
    output = await sandbox._exec(command="echo $MY_VAR")

    # In stateless mode, the variable should be empty (just a newline or empty)
    assert (
        "hello123" not in output
    ), f"Variable should NOT persist in stateless mode, got: {output}"

    print(f"Output (should be empty): '{output.strip()}'")
    print("✓ Stateless mode correctly does NOT persist state")

    sandbox._destroy()


async def test_stateful_mode_variable_persistence():
    """Test that stateful mode persists shell variables between commands."""
    print("\n=== Testing stateful mode (variable persistence) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Set a variable
    output1 = await sandbox._exec(command="export MY_VAR=hello123")
    print(f"Set variable output: '{output1}'")

    # Read it back - should work in stateful mode
    output2 = await sandbox._exec(command="echo $MY_VAR")
    print(f"Read variable output: '{output2}'")

    assert (
        "hello123" in output2
    ), f"Variable should persist in stateful mode, got: {output2}"

    print("✓ Stateful mode correctly persists variables")

    sandbox._destroy()


async def test_stateful_mode_cd_persistence():
    """Test that stateful mode persists working directory changes."""
    print("\n=== Testing stateful mode (cd persistence) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Create a directory and cd into it
    await sandbox._exec(command="mkdir -p /tmp/testdir")
    await sandbox._exec(command="cd /tmp/testdir")

    # Check pwd - should be in testdir
    output = await sandbox._exec(command="pwd")
    print(f"PWD output: '{output}'")

    assert "/tmp/testdir" in output, f"Directory should persist, got: {output}"

    print("✓ Stateful mode correctly persists working directory")

    sandbox._destroy()


async def test_stateful_mode_multiple_variables():
    """Test multiple variable operations in stateful mode."""
    print("\n=== Testing stateful mode (multiple variables) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Set multiple variables
    await sandbox._exec(command="export A=1")
    await sandbox._exec(command="export B=2")
    await sandbox._exec(command="export C=$((A + B))")

    # Read them back
    output = await sandbox._exec(command="echo A=$A B=$B C=$C")
    print(f"Variables output: '{output}'")

    assert "A=1" in output, f"A should be 1, got: {output}"
    assert "B=2" in output, f"B should be 2, got: {output}"
    assert "C=3" in output, f"C should be 3, got: {output}"

    print("✓ Stateful mode correctly handles multiple variables")

    sandbox._destroy()


async def test_stateful_mode_function_definition():
    """Test that shell functions persist in stateful mode."""
    print("\n=== Testing stateful mode (function definition) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Define a function
    await sandbox._exec(command='greet() { echo "Hello, $1!"; }')

    # Call the function
    output = await sandbox._exec(command="greet World")
    print(f"Function output: '{output}'")

    assert "Hello, World!" in output, f"Function should work, got: {output}"

    print("✓ Stateful mode correctly persists functions")

    sandbox._destroy()


async def test_stateful_mode_exit_codes():
    """Test that exit codes are correctly reported in stateful mode."""
    print("\n=== Testing stateful mode (exit codes) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Run a command that fails
    output = await sandbox._exec(command="ls /nonexistent_dir_12345")
    print(f"Failed command output: '{output}'")

    # Should include exit code indication
    assert (
        "Exit code:" in output or "No such file" in output or "cannot access" in output
    )

    # Run a command that succeeds
    output2 = await sandbox._exec(command="echo success")
    print(f"Success command output: '{output2}'")

    assert "success" in output2

    print("✓ Stateful mode correctly handles exit codes")

    sandbox._destroy()


async def test_stateful_mode_long_output():
    """Test handling of longer output in stateful mode."""
    print("\n=== Testing stateful mode (long output) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Generate longer output
    output = await sandbox._exec(
        command='for i in $(seq 1 100); do echo "Line $i: some text here"; done'
    )
    print(f"Long output (first 200 chars): '{output[:200]}...'")

    assert "Line 1:" in output
    assert "Line 100:" in output

    print("✓ Stateful mode correctly handles long output")

    sandbox._destroy()


async def test_stateful_mode_with_timeout():
    """Test that timeout works in stateful mode."""
    print("\n=== Testing stateful mode (timeout) ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True, stateful=True)

    # Quick command should work (timeout in ms)
    output = await sandbox._exec(command="echo quick", timeout=5000)
    assert "quick" in output

    print("✓ Stateful mode works with timeout")

    sandbox._destroy()


async def test_default_is_stateless():
    """Verify that the default mode is stateless."""
    print("\n=== Testing default mode is stateless ===")

    # Create without specifying stateful parameter
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    # Set a variable
    await sandbox._exec(command="export TEST_DEFAULT=xyz")

    # Try to read it
    output = await sandbox._exec(command="echo $TEST_DEFAULT")

    # Should NOT persist (default is stateless)
    assert "xyz" not in output, f"Default should be stateless, got: {output}"

    print("✓ Default mode is correctly stateless")

    sandbox._destroy()


async def test_new_parameters():
    """Test the new parameters: description, run_in_background, timeout in ms."""
    print("\n=== Testing new parameters ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    # Test description parameter (just ensures it doesn't error)
    output = await sandbox._exec(
        command="echo hello",
        description="Print hello message",
    )
    assert "hello" in output

    # Test run_in_background parameter
    result = await sandbox._exec(
        command="sleep 10",
        run_in_background=True,
        name="sleeper",
        description="Start background sleep process",
    )
    assert "Started background process" in result
    assert "sleeper" in result

    # Test timeout in milliseconds (default should work)
    output = await sandbox._exec(command="echo fast")
    assert "fast" in output

    print("✓ New parameters work correctly")

    sandbox._destroy()


async def main():
    """Run all stateful mode tests."""
    print("Testing ModalSandbox stateful mode...")

    # Test default behavior
    await test_default_is_stateless()

    # Stateless mode verification
    await test_stateless_mode_no_state_persistence()

    # Test new parameters
    await test_new_parameters()

    # Stateful mode tests
    await test_stateful_mode_variable_persistence()
    await test_stateful_mode_cd_persistence()
    await test_stateful_mode_multiple_variables()
    await test_stateful_mode_function_definition()
    await test_stateful_mode_exit_codes()
    await test_stateful_mode_long_output()
    await test_stateful_mode_with_timeout()

    print("\n✅ All stateful mode tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
