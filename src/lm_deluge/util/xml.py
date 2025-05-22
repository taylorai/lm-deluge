import re
from bs4 import BeautifulSoup, Tag

# import xml.etree.ElementTree as ET
from lxml import etree  # type: ignore


def get_tag(
    html_string: str, tag: str, return_attributes: bool = False
) -> dict | str | None:
    # Try to use regular expressions first
    if html_string is None:
        return None
    try:
        # Regex pattern to extract tag content and attributes
        pattern = re.compile(rf"<{tag}([^>]*)>(.*?)</{tag}>", re.DOTALL)
        match = pattern.search(html_string)
        if match:
            tag_attributes = match.group(1)  # Attributes string from the opening tag
            tag_contents = match.group(2)  # Contents inside the tag

            # If return_attributes is False, just return the tag contents
            if not return_attributes:
                return tag_contents

            # Parse attributes into a dictionary
            attributes_pattern = re.compile(
                r'(\w+)\s*=\s*"([^"]*)"'
            )  # Matches key="value"
            attributes = dict(attributes_pattern.findall(tag_attributes))

            return {"content": tag_contents, "attributes": attributes}
    except re.error:
        print(f"Failed to compile regular expression for HTML tag '{tag}'")

    # If regexp fails, use BeautifulSoup
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        tag_content = soup.find(tag)
        assert tag_content is None or isinstance(
            tag_content, Tag
        ), f"Unexpected type for tag_content: {type(tag_content)}"
        if tag_content is not None:
            tag_contents = tag_content.decode_contents()

            # If return_attributes is False, return just the content
            if not return_attributes:
                return tag_contents

            # Extract attributes from the tag
            attributes = tag_content.attrs

            return {"content": tag_contents, "attributes": attributes}
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
        # 1. find the tag + inner HTML exactly as before
        pattern = re.compile(rf"<{tag}([^>]*)>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(html_string)

        # 2. only parse attributes if the caller asked for them
        if not return_attributes:
            return [m[1] for m in matches]

        # --- new bits ---------------------------------------------------------------
        # key  = (\w+)
        # quote = (['"])          ← remembers whether it was ' or "
        # value = (.*?)\2         ← capture up to the *same* quote (back-ref \2)
        attributes_pattern = re.compile(r'(\w+)\s*=\s*([\'"])(.*?)\2')

        results = []
        for tag_attrs, tag_contents in matches:
            attrs = {key: val for key, _, val in attributes_pattern.findall(tag_attrs)}
            results.append({"content": tag_contents, "attributes": attrs})

        return results
    except re.error:
        print(f"Failed to compile regular expression for HTML tag '{tag}'")

    # Fallback to BeautifulSoup if regex fails
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        tag_contents = soup.find_all(tag)

        if not return_attributes:
            return [tag_content.decode_contents() for tag_content in tag_contents]  # type: ignore

        # Collect content and attributes when return_attributes is True
        results = []
        for tag_content in tag_contents:
            if isinstance(tag_content, Tag):
                results.append(
                    {
                        "content": tag_content.decode_contents(),
                        "attributes": tag_content.attrs,
                    }
                )
        return results
    except Exception as e:
        print(
            f"Failed to extract content from HTML tag '{tag}': {e}. Returning no matches."
        )

    return []


def strip_xml(xml_string: str) -> str:
    """
    Trim any text before the first '<' and after the last '>'.
    """
    if not xml_string:
        return ""
    start = xml_string.find("<")
    end = xml_string.rfind(">") + 1
    return xml_string[start:end] if start != -1 and end != 0 else xml_string


def remove_namespace_prefixes(xml_string):
    """
    Remove namespace prefixes from XML tags in the provided XML string.
    """
    # Remove namespace prefixes in opening and closing tags
    xml_string = re.sub(r"<(/?)(\w+:)", r"<\1", xml_string)
    # Remove namespace declarations in root element
    xml_string = re.sub(r'xmlns(:\w+)?="[^"]+"', "", xml_string)
    return xml_string


def object_to_xml(
    obj: dict | list | str | int | float,
    root_tag: str,
    ignore_dict_nulls: bool = True,
    list_item_tag: str = "li",  # could also be "option", "item", etc.
    include_list_index: bool = True,
    index_attr: str = "key",  # could be index, id, name, etc.
    indent_level: int = 0,
    indent_str: str = "  ",
    index=None,
):
    """
    Convert a Python object to an XML string.
    """
    xml = indent_str * indent_level
    xml += f"<{root_tag}"
    if include_list_index and index is not None:
        xml += f' {index_attr}="{index}"'
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
                indent_level=indent_level + 1,
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
                index=index,
            )
    else:
        raise ValueError("Unsupported object type.")

    xml += indent_str * indent_level
    xml += f"</{root_tag}>\n"
    return xml


def parse_base_element(
    elem_text: str | None, parse_empty_tags_as_none: bool, parse_null_text_as_none: bool
):
    if elem_text is None or elem_text.strip() == "":
        return None if parse_empty_tags_as_none else ""
    elem_text = elem_text.strip()
    if parse_null_text_as_none and elem_text.lower() == "null":
        return None
    # Try int first, then float
    try:
        return int(elem_text)
    except ValueError:
        try:
            return float(elem_text)
        except ValueError:
            return elem_text


def xml_to_object(
    xml_string: str, parse_null_text_as_none=True, parse_empty_tags_as_none=False
):
    """
    Intended to be the reverse of object_to_xml.
    Written by ChatGPT so unclear if this will work as intended.
    """
    xml_string = strip_xml(xml_string)
    xml_string = remove_namespace_prefixes(xml_string)
    parser = etree.XMLParser(recover=True, ns_clean=True)
    root = etree.fromstring(xml_string.encode("utf-8"), parser=parser)

    def parse_element(element):
        # Base case: element has no child elements
        if len(element) == 0:
            text = element.text
            return parse_base_element(
                text, parse_empty_tags_as_none, parse_null_text_as_none
            )
        else:
            is_list = False
            index_attrs = ["key", "index"]
            # Get child tag names without namespace prefixes
            child_tags = [etree.QName(child).localname for child in element]
            # Treat it as a list if there are any repeated children
            if (
                len(set(child_tags)) == 1
                and len(child_tags) == len(element)
                and len(child_tags) > 1
            ):
                is_list = True
            # Treat as list if it has one child, but the child has a "key" or "index" attribute
            elif len(child_tags) == 1 and any(
                attr in element[0].attrib for attr in index_attrs
            ):
                is_list = True
            # If multiple child tag types, but has repeats, error
            elif len(set(child_tags)) > 1 and len(set(child_tags)) < len(element):
                raise ValueError(
                    "Cannot parse XML with multiple child tags and repeats."
                )

            if is_list:
                items_with_index = []
                for child in element:
                    # look for either  <li key="…">  or  <li index="…">
                    index_value = None
                    for attr in ("key", "index"):
                        if attr in child.attrib:
                            index_value = child.attrib[attr]
                            break
                    # normalise to int when possible
                    try:
                        if index_value is not None:
                            index_value = int(index_value)
                    except ValueError:
                        pass

                    items_with_index.append((index_value, parse_element(child)))

                # Sort only when *all* items have an integer index
                if all(idx is not None for idx, _ in items_with_index):
                    items_with_index.sort(key=lambda x: x[0])
                return [item for _, item in items_with_index]
            else:
                # Treat as a dictionary
                obj = {}
                for child in element:
                    key = etree.QName(child).localname  # Get tag name without namespace
                    value = parse_element(child)
                    if key in obj:
                        raise ValueError(
                            f"Duplicate key '{key}' found in XML when not expecting a list."
                        )
                    obj[key] = value
                return obj

    return parse_element(root)
