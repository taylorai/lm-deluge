from __future__ import annotations

from typing import Any

from .core import Box2D, CoordinateSpace, Point, SpatialResult


def parse_gemini_boxes(data: list[dict[str, Any]]) -> SpatialResult:
    boxes: list[Box2D] = []
    for item in data:
        box = item.get("box_2d") or item.get("bbox")
        if not box:
            continue
        boxes.append(
            Box2D.from_list(
                box,
                order="yxyx",
                space=CoordinateSpace.NORMALIZED_1000,
            )
        )
    return SpatialResult(boxes=boxes, source="gemini", raw=data)


def parse_gemini_points(data: list[dict[str, Any]]) -> SpatialResult:
    points: list[Point] = []
    for item in data:
        point = item.get("point") or item.get("point_2d")
        if not point:
            continue
        points.append(
            Point(
                point[1],
                point[0],
                label=item.get("label"),
                description=item.get("description"),
                raw=item,
                source="gemini",
                space=CoordinateSpace.NORMALIZED_1000,
            )
        )
    return SpatialResult(points=points, source="gemini", raw=data)
