import re
from bs4 import BeautifulSoup, Tag
# import xml.etree.ElementTree as ET
from lxml import etree

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

def strip_xml(xml_string: str) -> str:
    """
    Remove all stuff outside of the XML so it can be parsed.
    """
    while not xml_string.startswith("<"):
        xml_string = xml_string[1:]
    while not xml_string.endswith(">"):
        xml_string = xml_string[:-1]

    # TODO: ensure it has a single root element
    return xml_string

def remove_namespace_prefixes(xml_string):
    """
    Remove namespace prefixes from XML tags in the provided XML string.
    """
    # Remove namespace prefixes in opening and closing tags
    xml_string = re.sub(r'<(/?)(\w+:)', r'<\1', xml_string)
    # Remove namespace declarations in root element
    xml_string = re.sub(r'xmlns(:\w+)?="[^"]+"', '', xml_string)
    return xml_string

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

def xml_to_object(
    xml_string: str,
    parse_null_text_as_none=True,
    empty_tags_as_none=False
):
    """
    Intended to be the reverse of object_to_xml.
    Written by ChatGPT so unclear if this will work as intended.
    """
    xml_string = strip_xml(xml_string)
    xml_string = remove_namespace_prefixes(xml_string)
    parser = etree.XMLParser(recover=True, ns_clean=True)
    root = etree.fromstring(xml_string.encode('utf-8'), parser=parser)

    def parse_element(element):
        # Base case: element has no child elements
        if len(element) == 0:
            text = element.text
            if text is None or text.strip() == '':
                return None if empty_tags_as_none else ''
            text = text.strip()
            if parse_null_text_as_none and text.lower() == 'null':
                return None
            # Try to convert text to int or float
            if text.isdigit():
                return int(text)
            else:
                try:
                    return float(text)
                except ValueError:
                    return text
        else:
            is_list = False
            index_attrs = ["key", "index"]
            # Get child tag names without namespace prefixes
            child_tags = [etree.QName(child).localname for child in element]
            # Treat it as a list if there are any repeated children
            if len(set(child_tags)) == 1 and len(child_tags) == len(element) and len(child_tags) > 1:
                is_list = True
            # Treat as list if it has one child, but the child has a "key" or "index" attribute
            elif len(child_tags) == 1 and any(attr in element[0].attrib for attr in index_attrs):
                is_list = True
            # If multiple child tag types, but has repeats, error
            elif len(set(child_tags)) > 1 and len(set(child_tags)) < len(element):
                raise ValueError("Cannot parse XML with multiple child tags and repeats.")

            if is_list:
                items_with_index = []
                for child in element:
                    index = child.attrib.get(index_attr)
                    if index is not None and index.isdigit():
                        index = int(index)
                    else:
                        index = None
                    item = parse_element(child)
                    items_with_index.append((index, item))
                # Sort items if indices are present
                if all(index is not None for index, _ in items_with_index):
                    items_with_index.sort(key=lambda x: x[0])
                items = [item for _, item in items_with_index]
                return items
            else:
                # Treat as a dictionary
                obj = {}
                for child in element:
                    key = etree.QName(child).localname  # Get tag name without namespace
                    value = parse_element(child)
                    if key in obj:
                        raise ValueError(f"Duplicate key '{key}' found in XML when not expecting a list.")
                    obj[key] = value
                return obj

    return parse_element(root)
