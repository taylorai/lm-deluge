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


def _escape_interior_quotes(json_string: str) -> str:
    """
    Escapes double quotes that appear inside JSON string values but were not
    properly escaped. Works by parsing character-by-character and using
    structural context to determine whether a quote is a real string
    delimiter or an interior quote that needs escaping.

    The heuristic: when we encounter a `"` that would close a string, we check
    if the next non-whitespace character is a valid JSON structural character
    (`,`, `}`, `]`, `:`). If not, this quote is interior to the string value
    and should be escaped. We also check the matching "opening" quote the same
    way — when a quote would open a new string, the preceding non-whitespace
    must be a structural char (`{`, `[`, `,`, `:`).
    """
    if not json_string:
        return json_string

    result = []
    i = 0
    in_string = False

    while i < len(json_string):
        c = json_string[i]

        if c == "\\" and in_string:
            # Escaped character — pass through both backslash and next char
            result.append(c)
            if i + 1 < len(json_string):
                i += 1
                result.append(json_string[i])
            i += 1
            continue

        if c == '"':
            if not in_string:
                # Opening quote — should always be valid here
                in_string = True
                result.append(c)
            else:
                # This quote would close the string. Check if it makes
                # structural sense as a closing quote.
                next_non_ws = _next_non_whitespace(json_string, i + 1)
                if next_non_ws in (",", "}", "]", ":", None):
                    # Valid close
                    in_string = False
                    result.append(c)
                else:
                    # This quote doesn't end the string structurally.
                    # Check if there's a matching close quote later on
                    # this same logical segment that DOES work.
                    # For now, escape it.
                    result.append('\\"')
            i += 1
            continue

        result.append(c)
        i += 1

    return "".join(result)


def _next_non_whitespace(s: str, start: int) -> str | None:
    """Return the next non-whitespace character at or after `start`, or None."""
    for i in range(start, len(s)):
        if not s[i].isspace():
            return s[i]
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

    # Try healing the JSON (trailing commas, missing brackets, etc.)
    if allow_healing:
        try:
            healed_json = heal_json(json_string)
            return json.loads(healed_json)
        except Exception:
            if allow_json5:
                try:
                    healed_json = heal_json(json_string)
                    return json5.loads(healed_json)
                except Exception:
                    pass

    # Try escaping unescaped interior quotes (more aggressive, so tried later)
    if allow_healing:
        try:
            escaped = _escape_interior_quotes(json_string)
            return json.loads(escaped)
        except Exception:
            pass
        # Try combining quote escaping with other healing
        escaped_and_healed = None
        try:
            escaped_and_healed = heal_json(_escape_interior_quotes(json_string))
            return json.loads(escaped_and_healed)
        except Exception:
            if allow_json5 and escaped_and_healed is not None:
                try:
                    return json5.loads(escaped_and_healed)
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
    except Exception:
        return None
