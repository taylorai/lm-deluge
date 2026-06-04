from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any

from .core import Box2D, CoordinateSpace, Point, Polygon, SpatialResult


TAG_PATTERN = re.compile(
    r"<(?P<tag>point|point_box|polygon|collection)\b(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)"
    r"</(?P=tag)>",
    re.DOTALL,
)
POINT_PATTERN = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def parse_perceptron(text: str) -> SpatialResult:
    points: list[Point] = []
    boxes: list[Box2D] = []
    polygons: list[Polygon] = []

    for match in TAG_PATTERN.finditer(text):
        tag = match.group("tag")
        attrs = _parse_attrs(match.group("attrs"))
        body = html.unescape(match.group("body")).strip()
        coords = _parse_points(body)
        label = attrs.get("mention") or attrs.get("label") or attrs.get("name")
        t = _optional_float(attrs.get("t"))

        if tag == "collection":
            nested = parse_perceptron(body)
            points.extend(nested.points)
            boxes.extend(nested.boxes)
            polygons.extend(nested.polygons)
        elif tag == "point":
            for x, y in coords:
                points.append(
                    Point(
                        x,
                        y,
                        label=label,
                        description=label,
                        t=t,
                        source="perceptron",
                        raw=match.group(0),
                        extra=attrs,
                        space=CoordinateSpace.NORMALIZED_1000,
                    )
                )
        elif tag == "point_box" and len(coords) >= 2:
            (x1, y1), (x2, y2) = coords[0], coords[1]
            boxes.append(
                Box2D(
                    x1,
                    y1,
                    x2,
                    y2,
                    label=label,
                    description=label,
                    t=t,
                    source="perceptron",
                    raw=match.group(0),
                    extra=attrs,
                    space=CoordinateSpace.NORMALIZED_1000,
                )
            )
        elif tag == "polygon" and coords:
            polygons.append(
                Polygon(
                    [
                        Point(
                            x,
                            y,
                            label=label,
                            description=label,
                            t=t,
                            source="perceptron",
                            raw=match.group(0),
                            extra=attrs,
                            space=CoordinateSpace.NORMALIZED_1000,
                        )
                        for x, y in coords
                    ],
                    label=label,
                    description=label,
                    t=t,
                    source="perceptron",
                    raw=match.group(0),
                    extra=attrs,
                )
            )

    text_without_tags = TAG_PATTERN.sub("", text).strip() or None
    return SpatialResult(
        points=points,
        boxes=boxes,
        polygons=polygons,
        text=text_without_tags,
        source="perceptron",
        raw=text,
    )


def _parse_attrs(raw_attrs: str) -> dict[str, Any]:
    if not raw_attrs.strip():
        return {}
    try:
        element = ET.fromstring(f"<tag {raw_attrs}></tag>")
    except ET.ParseError:
        return {}
    return dict(element.attrib)


def _parse_points(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in POINT_PATTERN.findall(text)]


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
