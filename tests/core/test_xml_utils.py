from __future__ import annotations

from lm_deluge.util.xml import (
    xml_to_object,
    object_to_xml,
    strip_xml,
    get_tag,
    get_tags,
)


def test_round_trip_object_to_xml():
    data = {"foo": "bar", "nums": [1, 2]}
    xml = object_to_xml(data, "root")
    assert xml_to_object(xml) == data


def test_get_tag_and_get_tags():
    xml = "<root><name a='1'>first</name><name>second</name></root>"
    assert get_tag(xml, "name") == "first"
    tags = get_tags(xml, "name", return_attributes=True)
    # print(len(tags), "tags found:", tags)
    assert tags[0]["attributes"]["a"] == "1"
    assert tags[1]["content"] == "second"


def test_strip_xml():
    xml = "prefix <data>val</data> suffix"
    assert strip_xml(xml) == "<data>val</data>"


if __name__ == "__main__":
    test_round_trip_object_to_xml()
    test_get_tag_and_get_tags()
    test_strip_xml()
