import json
import re

import json5


def extract_quoted_expressions(json_string: str):
    # This pattern matches double-quoted strings while handling escaped quotes
    pattern = r'"((?:\\.|[^"\\])*)"'
    expressions = re.findall(pattern, json_string, re.DOTALL)
    return expressions


def string_list_to_dict(string_list: list[str]) -> dict:
    """
    Converts a list of strings to a dictionary.
    The list should contain alternating keys and values.
    """
    result_dict = {}
    # Iterate over the list in steps of 2 to get key-value pairs
    for i in range(0, len(string_list) - 1, 2):
        key = string_list[i]
        value = string_list[i + 1]
        result_dict[key] = value
    return result_dict


def strip_json(json_string: str | None) -> str | None:
    """
    Strips extra stuff from beginning & end of JSON string.
    """
    if json_string is None:
        return None
    json_string = json_string.strip()
    json_string = json_string.removeprefix("```json")
    json_string = json_string.removesuffix("```")
    if "```json\n" in json_string:
        json_string = json_string.split("```json\n", 1)[1]
    json_string = json_string.strip("`").strip()

    # not strict enough!
    if "[" not in json_string and "{" not in json_string:
        return None

    # Find the first opening bracket/brace
    start_idx = min(
        (json_string.find("{") if "{" in json_string else len(json_string)),
        (json_string.find("[") if "[" in json_string else len(json_string)),
    )

    # Find the last closing bracket/brace
    end_idx = max(json_string.rfind("}"), json_string.rfind("]"))

    if start_idx >= 0 and end_idx >= 0:
        return json_string[start_idx : end_idx + 1]

    return None


def heal_json(json_string: str) -> str:
    """
    Attempts to heal malformed JSON by fixing common issues like unclosed brackets and braces.
    Uses a stack-based approach to ensure proper nesting order is maintained.

    :param json_string: The potentially malformed JSON string
    :return: A hopefully valid JSON string
    """
    if not json_string:
        return json_string

    # Handle trailing commas before closing brackets
    json_string = re.sub(r",\s*}", "}", json_string)
    json_string = re.sub(r",\s*\]", "]", json_string)

    # Use a stack to track opening brackets
    stack = []
    for char in json_string:
        if char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()

    # Add missing closing braces/brackets in the correct order
    closing = ""
    while stack:
        bracket = stack.pop()
        if bracket == "{":
            closing += "}"
        elif bracket == "[":
            closing += "]"

    # Check for unclosed strings
    quote_count = json_string.count('"') - json_string.count('\\"')
    if quote_count % 2 == 1:  # Odd number of quotes means unclosed string
        # Find the last unescaped quote
        last_pos = -1
        i = 0
        while i < len(json_string):
            if json_string[i] == '"' and (i == 0 or json_string[i - 1] != "\\"):
                last_pos = i
            i += 1

        # If we have unclosed quotes and the last one is not followed by closing brackets
        if last_pos != -1 and last_pos < len(json_string) - 1:
            # Add closing quote before any closing brackets we're going to add
            json_string += '"'

    return json_string + closing


def load_json(
    json_string: str | None,
    allow_json5: bool = True,
    allow_partial: bool = False,
    allow_healing: bool = True,
):
    """
    Loads a JSON string into a Python object.
    :param json_string: The JSON string to load.
    :param allow_json5: Whether to allow lax parsing of the JSON string.
    :param allow_partial: Whether to allow partial parsing of the JSON string.
    This will extract as many valid fields as possible.
    :param allow_healing: Whether to attempt to heal malformed JSON.
    :return: The loaded Python object.
    """
    if json_string is None:
        raise ValueError("Invalid (None) json_string")
    json_string = strip_json(json_string)
    if json_string is None or len(json_string) == 0:
        raise ValueError("Invalid (empty) json_string")

    # Try standard JSON parsing
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    # Try JSON5 parsing
    if allow_json5:
        try:
            return json5.loads(json_string)
        except Exception:
            pass

    # Try healing the JSON
    if allow_healing:
        try:
            healed_json = heal_json(json_string)
            return json.loads(healed_json)
        except Exception:
            # If healing with standard JSON fails, try with JSON5
            if allow_json5:
                try:
                    healed_json = heal_json(json_string)
                    return json5.loads(healed_json)
                except Exception:
                    pass

    # Try partial parsing as a last resort
    if allow_partial:
        try:
            string_list = extract_quoted_expressions(json_string)
            return string_list_to_dict(string_list)
        except Exception:
            pass

    raise ValueError(f"Invalid JSON string: {json_string}")


def try_load_json(
    json_string: str | None,
    allow_json5: bool = True,
    allow_partial: bool = False,
    allow_healing: bool = True,
):
    """
    Like the above, except it returns None instead of raising an error.
    """
    try:
        return load_json(json_string, allow_json5, allow_partial, allow_healing)
    except Exception as e:
        print(f"Failed to load json: {e}. Returning None.")
        return None
