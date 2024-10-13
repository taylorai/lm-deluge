import re
import json
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
    if json_string.startswith("```json"):
        json_string = json_string.split("```json", 1)[1]
    json_string = json_string.strip("`")

    # not strict enough!
    while not json_string.startswith("{") and not json_string.startswith("["):
        json_string = json_string[1:]
    while not json_string.endswith("}") and not json_string.endswith("]"):
        json_string = json_string[:-1]

    return json_string

def load_json(
    json_string: str | None,
    allow_json5: bool = True,
    allow_partial: bool = False
):
    """
    Loads a JSON string into a Python object.
    :param json_string: The JSON string to load.
    :param allow_lax: Whether to allow lax parsing of the JSON string.
    :param allow_partial: Whether to allow partial parsing of the JSON string.
    This will extract as many valid fields as possible.
    :return: The loaded Python object.
    """
    if json_string is None:
        return None
    json_string = strip_json(json_string)
    assert json_string is not None, "json_string is None"
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    if allow_json5:
        try:
            return json5.loads(json_string)
        except Exception as e:
            pass
    if allow_partial:
        try:
            string_list = extract_quoted_expressions(json_string)
            return string_list_to_dict(string_list)
        except Exception:
            pass
    raise ValueError(f"Invalid JSON string: {json_string}")
