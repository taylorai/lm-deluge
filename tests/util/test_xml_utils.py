from __future__ import annotations

import types
import sys

import pytest
from tests.helpers import import_module

# Stub PIL since some modules may expect it
pil_stub = types.ModuleType("PIL")
pil_stub.Image = type("Image", (), {})
sys.modules.setdefault("PIL", pil_stub)

xml_utils = import_module("src/llm_utils/util/xml.py", name="xml_utils")


def test_round_trip_object_to_xml():
    data = {"foo": "bar", "nums": [1, 2]}
    xml = xml_utils.object_to_xml(data, "root")
    assert xml_utils.xml_to_object(xml) == data


def test_get_tag_and_get_tags():
    xml = "<root><name a='1'>first</name><name>second</name></root>"
    assert xml_utils.get_tag(xml, "name") == "first"
    tags = xml_utils.get_tags(xml, "name", return_attributes=True)
    assert tags[0]["attributes"]["a"] == "1"
    assert tags[1]["content"] == "second"


def test_strip_xml():
    xml = "prefix <data>val</data> suffix"
    assert xml_utils.strip_xml(xml) == "<data>val</data>"
