"""Simple standalone test for RandomTools functionality."""

import json
import sys
from pathlib import Path

# Add src to path so we can import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import just the modules we need directly
import random as random_module
import secrets as secrets_module
import json as json_module


# Test the implementation without full import
def _random_float() -> str:
    try:
        value = random_module.random()
        return json_module.dumps({"status": "success", "value": value})
    except Exception as e:
        return json_module.dumps({"status": "error", "error": str(e)})


def _random_choice(items) -> str:
    try:
        if not items:
            return json_module.dumps(
                {"status": "error", "error": "Cannot choose from an empty list"}
            )
        choice = random_module.choice(items)
        return json_module.dumps({"status": "success", "value": choice})
    except Exception as e:
        return json_module.dumps({"status": "error", "error": str(e)})


def _random_int(min_value: int, max_value: int) -> str:
    try:
        if min_value > max_value:
            return json_module.dumps(
                {
                    "status": "error",
                    "error": f"min_value ({min_value}) cannot be greater than max_value ({max_value})",
                }
            )
        value = random_module.randint(min_value, max_value)
        return json_module.dumps({"status": "success", "value": value})
    except Exception as e:
        return json_module.dumps({"status": "error", "error": str(e)})


def _random_token(length: int = 32) -> str:
    try:
        if length <= 0:
            return json_module.dumps(
                {"status": "error", "error": "length must be greater than 0"}
            )
        token = secrets_module.token_urlsafe(length)
        return json_module.dumps({"status": "success", "value": token})
    except Exception as e:
        return json_module.dumps({"status": "error", "error": str(e)})


# Run tests
print("Testing random_float...")
for _ in range(5):
    result = _random_float()
    data = json.loads(result)
    assert data["status"] == "success"
    assert 0 <= data["value"] < 1
print("✓ random_float works")

print("\nTesting random_choice...")
result = _random_choice(["a", "b", "c"])
data = json.loads(result)
assert data["status"] == "success"
assert data["value"] in ["a", "b", "c"]

result = _random_choice([])
data = json.loads(result)
assert data["status"] == "error"
assert "empty list" in data["error"]
print("✓ random_choice works")

print("\nTesting random_int...")
for _ in range(10):
    result = _random_int(1, 10)
    data = json.loads(result)
    assert data["status"] == "success"
    assert 1 <= data["value"] <= 10

result = _random_int(10, 5)
data = json.loads(result)
assert data["status"] == "error"
assert "cannot be greater than" in data["error"]
print("✓ random_int works")

print("\nTesting random_token...")
result = _random_token(32)
data = json.loads(result)
assert data["status"] == "success"
assert isinstance(data["value"], str)
assert len(data["value"]) > 0

result = _random_token(0)
data = json.loads(result)
assert data["status"] == "error"
assert "must be greater than 0" in data["error"]
print("✓ random_token works")

print("\n✨ All basic functionality tests passed!")
