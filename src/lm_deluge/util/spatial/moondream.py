from __future__ import annotations

from typing import Any

from .core import Box2D, CoordinateSpace, Mask, Point, SpatialResult


def parse_moondream(data: dict[str, Any]) -> SpatialResult:
    points = [
        Point.from_dict(point, space=CoordinateSpace.NORMALIZED)
        for point in data.get("points", [])
    ]
    boxes = [
        Box2D.from_dict(box, space=CoordinateSpace.NORMALIZED)
        for box in data.get("objects", [])
    ]

    mask = None
    if "path" in data:
        bbox = None
        if isinstance(data.get("bbox"), dict):
            bbox = Box2D.from_dict(data["bbox"], space=CoordinateSpace.NORMALIZED)
        mask = Mask(
            path=data["path"],
            bbox=bbox,
            source="moondream",
            raw=data,
        )

    text = data.get("answer") or data.get("caption")
    return SpatialResult(
        points=points,
        boxes=boxes,
        masks=[mask] if mask else [],
        text=text,
        source="moondream",
        raw=data,
    )
