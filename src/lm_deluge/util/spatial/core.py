from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class CoordinateSpace(str, Enum):
    PIXELS = "pixels"
    NORMALIZED = "normalized"
    NORMALIZED_1000 = "normalized_1000"


@dataclass(slots=True)
class SpatialMetadata:
    label: str | None = None
    description: str | None = None
    confidence: float | None = None
    frame: int | None = None
    t: float | None = None
    source: str | None = None
    raw: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Point:
    x: float
    y: float
    label: str | None = None
    description: str | None = None
    confidence: float | None = None
    frame: int | None = None
    t: float | None = None
    source: str | None = None
    raw: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    space: CoordinateSpace = CoordinateSpace.NORMALIZED

    def __getitem__(self, index):
        if index in {0, "x"}:
            return self.x
        if index in {1, "y"}:
            return self.y
        raise IndexError("Index out of range")

    def __iter__(self):
        yield self.x
        yield self.y

    @classmethod
    def from_tuple(
        cls,
        point: tuple[float, float],
        description: str | None = None,
        *,
        space: CoordinateSpace = CoordinateSpace.NORMALIZED,
    ) -> "Point":
        return cls(point[0], point[1], description=description, space=space)

    @classmethod
    def from_dict(
        cls,
        point: dict[str, Any],
        description: str | None = None,
        *,
        space: CoordinateSpace = CoordinateSpace.NORMALIZED,
    ) -> "Point":
        return cls(
            point["x"],
            point["y"],
            label=point.get("label") or point.get("name"),
            description=description or point.get("description"),
            confidence=point.get("confidence"),
            frame=point.get("frame"),
            t=point.get("t"),
            raw=point,
            space=space,
        )

    @property
    def metadata(self) -> SpatialMetadata:
        return SpatialMetadata(
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
        )

    def with_space(self, space: CoordinateSpace) -> "Point":
        return self.__class__(
            self.x,
            self.y,
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
            space=space,
        )

    def scale_to(
        self,
        dst_width: float,
        dst_height: float,
        *,
        src_width: float | None = None,
        src_height: float | None = None,
        space: CoordinateSpace = CoordinateSpace.PIXELS,
        return_integers: bool = False,
    ) -> "Point":
        src_width, src_height = _resolve_source_dimensions(
            self.space, src_width, src_height
        )
        x = self.x * dst_width / src_width
        y = self.y * dst_height / src_height
        if return_integers:
            x = int(x)
            y = int(y)
        return self.__class__(
            x,
            y,
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
            space=space,
        )

    def to_pixels(
        self, width: int, height: int, *, return_integers: bool = True
    ) -> "Point":
        return self.scale_to(
            width,
            height,
            space=CoordinateSpace.PIXELS,
            return_integers=return_integers,
        )

    def to_normalized(self) -> "Point":
        return self.scale_to(1, 1, space=CoordinateSpace.NORMALIZED)

    def to_1000(self, *, return_integers: bool = True) -> "Point":
        return self.scale_to(
            1000,
            1000,
            space=CoordinateSpace.NORMALIZED_1000,
            return_integers=return_integers,
        )


@dataclass(slots=True)
class Box2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str | None = None
    description: str | None = None
    confidence: float | None = None
    frame: int | None = None
    t: float | None = None
    source: str | None = None
    raw: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    space: CoordinateSpace = CoordinateSpace.NORMALIZED

    @property
    def x_min(self) -> float:
        return self.xmin

    @property
    def y_min(self) -> float:
        return self.ymin

    @property
    def x_max(self) -> float:
        return self.xmax

    @property
    def y_max(self) -> float:
        return self.ymax

    @classmethod
    def from_list(
        cls,
        box: list[float],
        *,
        order: Literal["yxyx", "xyxy"] = "yxyx",
        space: CoordinateSpace = CoordinateSpace.NORMALIZED,
    ) -> "Box2D":
        if order == "yxyx":
            return cls(box[1], box[0], box[3], box[2], space=space)
        return cls(box[0], box[1], box[2], box[3], space=space)

    @classmethod
    def from_dict(
        cls,
        box: dict[str, Any],
        *,
        space: CoordinateSpace = CoordinateSpace.NORMALIZED,
    ) -> "Box2D":
        xmin = box.get("xmin", box.get("x_min"))
        ymin = box.get("ymin", box.get("y_min"))
        xmax = box.get("xmax", box.get("x_max"))
        ymax = box.get("ymax", box.get("y_max"))
        if xmin is None or ymin is None or xmax is None or ymax is None:
            raise ValueError(f"Invalid box dictionary keys: {box}")
        return cls(
            float(xmin),
            float(ymin),
            float(xmax),
            float(ymax),
            label=box.get("label") or box.get("name"),
            description=box.get("description"),
            confidence=box.get("confidence"),
            frame=box.get("frame"),
            t=box.get("t"),
            raw=box,
            space=space,
        )

    def center(self) -> Point:
        return Point(
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
            space=self.space,
        )

    def with_space(self, space: CoordinateSpace) -> "Box2D":
        return self.__class__(
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
            space=space,
        )

    def scale_to(
        self,
        dst_width: float,
        dst_height: float,
        *,
        src_width: float | None = None,
        src_height: float | None = None,
        space: CoordinateSpace = CoordinateSpace.PIXELS,
        return_integers: bool = False,
    ) -> "Box2D":
        src_width, src_height = _resolve_source_dimensions(
            self.space, src_width, src_height
        )
        xmin = self.xmin * dst_width / src_width
        ymin = self.ymin * dst_height / src_height
        xmax = self.xmax * dst_width / src_width
        ymax = self.ymax * dst_height / src_height
        if return_integers:
            xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))
        return self.__class__(
            xmin,
            ymin,
            xmax,
            ymax,
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
            space=space,
        )

    def to_pixels(
        self, width: int, height: int, *, return_integers: bool = True
    ) -> "Box2D":
        return self.scale_to(
            width,
            height,
            space=CoordinateSpace.PIXELS,
            return_integers=return_integers,
        )

    def to_normalized(self) -> "Box2D":
        return self.scale_to(1, 1, space=CoordinateSpace.NORMALIZED)

    def to_1000(self, *, return_integers: bool = True) -> "Box2D":
        return self.scale_to(
            1000,
            1000,
            space=CoordinateSpace.NORMALIZED_1000,
            return_integers=return_integers,
        )


@dataclass(slots=True)
class Polygon:
    points: list[Point]
    label: str | None = None
    description: str | None = None
    confidence: float | None = None
    frame: int | None = None
    t: float | None = None
    source: str | None = None
    raw: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def space(self) -> CoordinateSpace:
        if not self.points:
            return CoordinateSpace.NORMALIZED
        return self.points[0].space

    def to_pixels(
        self, width: int, height: int, *, return_integers: bool = True
    ) -> "Polygon":
        return self.__class__(
            [
                p.to_pixels(width, height, return_integers=return_integers)
                for p in self.points
            ],
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
        )

    def to_normalized(self) -> "Polygon":
        return self.__class__(
            [p.to_normalized() for p in self.points],
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
        )

    def to_1000(self, *, return_integers: bool = True) -> "Polygon":
        return self.__class__(
            [p.to_1000(return_integers=return_integers) for p in self.points],
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            frame=self.frame,
            t=self.t,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
        )


@dataclass(slots=True)
class Mask:
    path: str
    bbox: Box2D | None = None
    label: str | None = None
    description: str | None = None
    confidence: float | None = None
    source: str | None = None
    raw: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_pixels(
        self, width: int, height: int, *, return_integers: bool = True
    ) -> "Mask":
        return self.__class__(
            path=self.path,
            bbox=(
                self.bbox.to_pixels(width, height, return_integers=return_integers)
                if self.bbox
                else None
            ),
            label=self.label,
            description=self.description,
            confidence=self.confidence,
            source=self.source,
            raw=self.raw,
            extra=dict(self.extra),
        )


@dataclass(slots=True)
class SpatialResult:
    points: list[Point] = field(default_factory=list)
    boxes: list[Box2D] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    text: str | None = None
    source: str | None = None
    raw: Any | None = None

    def __bool__(self) -> bool:
        return bool(
            self.points or self.boxes or self.polygons or self.masks or self.text
        )

    def __len__(self) -> int:
        return len(self.points) + len(self.boxes) + len(self.polygons) + len(self.masks)

    def to_pixels(
        self, width: int, height: int, *, return_integers: bool = True
    ) -> "SpatialResult":
        return self.__class__(
            points=[
                p.to_pixels(width, height, return_integers=return_integers)
                for p in self.points
            ],
            boxes=[
                b.to_pixels(width, height, return_integers=return_integers)
                for b in self.boxes
            ],
            polygons=[
                p.to_pixels(width, height, return_integers=return_integers)
                for p in self.polygons
            ],
            masks=[
                m.to_pixels(width, height, return_integers=return_integers)
                for m in self.masks
            ],
            text=self.text,
            source=self.source,
            raw=self.raw,
        )

    def to_normalized(self) -> "SpatialResult":
        return self.__class__(
            points=[p.to_normalized() for p in self.points],
            boxes=[b.to_normalized() for b in self.boxes],
            polygons=[p.to_normalized() for p in self.polygons],
            masks=self.masks,
            text=self.text,
            source=self.source,
            raw=self.raw,
        )

    def to_1000(self, *, return_integers: bool = True) -> "SpatialResult":
        return self.__class__(
            points=[p.to_1000(return_integers=return_integers) for p in self.points],
            boxes=[b.to_1000(return_integers=return_integers) for b in self.boxes],
            polygons=[
                p.to_1000(return_integers=return_integers) for p in self.polygons
            ],
            masks=self.masks,
            text=self.text,
            source=self.source,
            raw=self.raw,
        )


SpatialObject = Point | Box2D | Polygon | Mask


def scale(
    obj: Point | Box2D | tuple[float, ...] | dict[str, Any],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    return_integers: bool = True,
) -> Point | Box2D:
    spatial_obj: Point | Box2D
    if isinstance(obj, tuple):
        if len(obj) == 2:
            spatial_obj = Point(obj[0], obj[1], space=CoordinateSpace.PIXELS)
        elif len(obj) == 4:
            spatial_obj = Box2D(
                obj[0], obj[1], obj[2], obj[3], space=CoordinateSpace.PIXELS
            )
        else:
            raise ValueError("Invalid tuple length")
    elif isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            spatial_obj = Point.from_dict(obj, space=CoordinateSpace.PIXELS)
        else:
            spatial_obj = Box2D.from_dict(obj, space=CoordinateSpace.PIXELS)
    else:
        spatial_obj = obj

    return spatial_obj.scale_to(
        dst_width,
        dst_height,
        src_width=src_width,
        src_height=src_height,
        space=CoordinateSpace.PIXELS,
        return_integers=return_integers,
    )


def normalize_point(
    obj: Point | Box2D | tuple[float, ...] | dict[str, Any],
    src_width: int,
    src_height: int,
) -> Point | Box2D:
    return scale(
        obj,
        src_width=src_width,
        src_height=src_height,
        dst_width=1,
        dst_height=1,
        return_integers=False,
    ).with_space(CoordinateSpace.NORMALIZED)


def denormalize_point(
    obj: Point | Box2D | tuple[float, ...] | dict[str, Any],
    dst_width: int,
    dst_height: int,
) -> Point | Box2D:
    if isinstance(obj, (Point, Box2D)):
        return obj.to_pixels(dst_width, dst_height)
    if isinstance(obj, tuple) and len(obj) == 2:
        return Point(obj[0], obj[1]).to_pixels(dst_width, dst_height)
    if isinstance(obj, tuple) and len(obj) == 4:
        return Box2D(obj[0], obj[1], obj[2], obj[3]).to_pixels(dst_width, dst_height)
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            return Point.from_dict(obj).to_pixels(dst_width, dst_height)
        return Box2D.from_dict(obj).to_pixels(dst_width, dst_height)
    raise ValueError("Invalid tuple length")


def normalize_point_1k(
    obj: Point | Box2D | tuple[float, ...] | dict[str, Any],
    src_width: int,
    src_height: int,
) -> Point | Box2D:
    return scale(
        obj,
        src_width=src_width,
        src_height=src_height,
        dst_width=1000,
        dst_height=1000,
        return_integers=True,
    ).with_space(CoordinateSpace.NORMALIZED_1000)


def denormalize_point_1k(
    obj: Point | Box2D | tuple[float, ...] | dict[str, Any],
    dst_width: int,
    dst_height: int,
) -> Point | Box2D:
    if isinstance(obj, (Point, Box2D)):
        return obj.scale_to(
            dst_width,
            dst_height,
            src_width=1000,
            src_height=1000,
            space=CoordinateSpace.PIXELS,
            return_integers=False,
        )
    if isinstance(obj, tuple) and len(obj) == 2:
        return Point(obj[0], obj[1], space=CoordinateSpace.NORMALIZED_1000).to_pixels(
            dst_width, dst_height, return_integers=False
        )
    if isinstance(obj, tuple) and len(obj) == 4:
        return Box2D(
            obj[0], obj[1], obj[2], obj[3], space=CoordinateSpace.NORMALIZED_1000
        ).to_pixels(dst_width, dst_height, return_integers=False)
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            return Point.from_dict(
                obj, space=CoordinateSpace.NORMALIZED_1000
            ).to_pixels(dst_width, dst_height, return_integers=False)
        return Box2D.from_dict(
            obj,
            space=CoordinateSpace.NORMALIZED_1000,
        ).to_pixels(dst_width, dst_height, return_integers=False)
    raise ValueError("Invalid tuple length")


def _resolve_source_dimensions(
    space: CoordinateSpace,
    src_width: float | None,
    src_height: float | None,
) -> tuple[float, float]:
    if src_width is not None and src_height is not None:
        return src_width, src_height
    if space == CoordinateSpace.NORMALIZED:
        return 1, 1
    if space == CoordinateSpace.NORMALIZED_1000:
        return 1000, 1000
    raise ValueError("Pixel-space objects require src_width and src_height")
