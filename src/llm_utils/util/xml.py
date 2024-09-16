import re
from bs4 import BeautifulSoup, Tag

def get_tag(html_string: str, tag: str, return_attributes: bool = False) -> dict | str | None:
    # Try to use regular expressions first
    if html_string is None:
        return None
    try:
        # Regex pattern to extract tag content and attributes
        pattern = re.compile(rf"<{tag}([^>]*)>(.*?)</{tag}>", re.DOTALL)
        match = pattern.search(html_string)
        if match:
            tag_attributes = match.group(1)  # Attributes string from the opening tag
            tag_contents = match.group(2)    # Contents inside the tag

            # If return_attributes is False, just return the tag contents
            if not return_attributes:
                return tag_contents

            # Parse attributes into a dictionary
            attributes_pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')  # Matches key="value"
            attributes = dict(attributes_pattern.findall(tag_attributes))

            return {
                "content": tag_contents,
                "attributes": attributes
            }
    except re.error:
        print(f"Failed to compile regular expression for HTML tag '{tag}'")

    # If regexp fails, use BeautifulSoup
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        tag_content = soup.find(tag)
        assert tag_content is None or isinstance(tag_content, Tag), f"Unexpected type for tag_content: {type(tag_content)}"
        if tag_content is not None:
            tag_contents = tag_content.decode_contents()

            # If return_attributes is False, return just the content
            if not return_attributes:
                return tag_contents

            # Extract attributes from the tag
            attributes = tag_content.attrs

            return {
                "content": tag_contents,
                "attributes": attributes
            }
    except Exception as e:
        print(f"Failed to extract content from HTML tag '{tag}': {e}. Returning None.")

    return None


def get_tags(html_string: str, tag: str, return_attributes: bool = False) -> list:
    """
    Extract all instances of the <tag></tag> in the string, not just the first.
    If return_attributes is True, also return the tag's attributes.
    """
    if html_string is None:
        return []

    try:
        # Regex pattern to match all instances of the tag and capture attributes and content
        pattern = re.compile(rf"<{tag}([^>]*)>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(html_string)

        if not return_attributes:
            return [match[1] for match in matches]  # Return just the content inside the tags

        # Parse attributes if return_attributes is True
        attributes_pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')  # Matches key="value"

        results = []
        for match in matches:
            tag_attributes = match[0]  # The attributes portion of the tag
            tag_contents = match[1]    # The content portion of the tag

            # Parse attributes into a dictionary
            attributes = dict(attributes_pattern.findall(tag_attributes))
            results.append({
                "content": tag_contents,
                "attributes": attributes
            })
        return results
    except re.error:
        print(f"Failed to compile regular expression for HTML tag '{tag}'")

    # Fallback to BeautifulSoup if regex fails
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        tag_contents = soup.find_all(tag)

        if not return_attributes:
            return [tag_content.decode_contents() for tag_content in tag_contents]

        # Collect content and attributes when return_attributes is True
        results = []
        for tag_content in tag_contents:
            if isinstance(tag_content, Tag):
                results.append({
                    "content": tag_content.decode_contents(),
                    "attributes": tag_content.attrs
                })
        return results
    except Exception as e:
        print(f"Failed to extract content from HTML tag '{tag}': {e}. Returning no matches.")

    return []

def object_to_xml(
    obj: dict | list | str | int | float,
    root_tag: str,
    ignore_dict_nulls: bool = True,
    list_item_tag: str = "li", # could also be "option", "item", etc.
    include_list_index: bool = True,
    index_attr: str = "key", # could be index, id, name, etc.
    indent_level: int = 0,
    indent_str: str = "  ",
    index=None
):
    """
    Convert a Python object to an XML string.
    """
    xml = indent_str * indent_level
    xml += f"<{root_tag}"
    if include_list_index and index is not None:
        xml += f" {index_attr}=\"{index}\""
    xml += ">\n"
    # base case
    if isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float):
        xml += indent_str * (indent_level + 1)
        xml += f"{obj}\n"
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if ignore_dict_nulls and value is None:
                continue
            xml += object_to_xml(
                value,
                root_tag=key,
                list_item_tag=list_item_tag,
                include_list_index=include_list_index,
                index_attr=index_attr,
                indent_level=indent_level + 1
            )
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            xml += object_to_xml(
                item,
                root_tag=list_item_tag,
                list_item_tag=list_item_tag,
                include_list_index=include_list_index,
                index_attr=index_attr,
                indent_level=indent_level + 1,
                index=index
            )
    else:
        raise ValueError("Unsupported object type.")

    xml += indent_str * indent_level
    xml += f"</{root_tag}>\n"
    return xml
