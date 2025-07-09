from dataclasses import dataclass
from functools import partial

from PIL import Image, ImageDraw


@dataclass
class Point:
    def __init__(self, x: float, y: float, description: str | None = None):
        self.x = x
        self.y = y
        self.description = description

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == "x":
            return self.x
        elif index == "y":
            return self.y
        else:
            raise IndexError("Index out of range")

    def __iter__(self):
        yield self.x
        yield self.y

    def __str__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    @classmethod
    def from_tuple(cls, point: tuple, description: str | None = None):
        return cls(point[0], point[1], description)

    @classmethod
    def from_dict(cls, point: dict, description: str | None = None):
        return cls(point["x"], point["y"], description)


@dataclass
class Box2D:
    def __init__(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        description: str | None = None,
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.description = description

    def __repr__(self):
        return f"Box2D(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"

    def __str__(self):
        return f"Box2D(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"

    @classmethod
    def from_list(cls, box: list):
        return cls(box[1], box[0], box[3], box[2])

    def center(self):
        return Point((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)


def scale(
    obj: Point | Box2D | tuple | dict,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    return_integers: bool = True,
):
    if isinstance(obj, tuple):
        if len(obj) == 2:
            obj = Point(obj[0], obj[1])
        elif len(obj) == 4:
            obj = Box2D(obj[0], obj[1], obj[2], obj[3])
        else:
            raise ValueError("Invalid tuple length")
    elif isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            obj = Point(obj["x"], obj["y"])
        elif "xmin" in obj and "ymin" in obj and "xmax" in obj and "ymax" in obj:
            obj = Box2D(obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
        else:
            raise ValueError("Invalid dictionary keys")
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height

    if isinstance(obj, Point):
        x = obj.x * scale_x
        y = obj.y * scale_y
        if return_integers:
            return Point(int(x), int(y), obj.description)
        return Point(x, y)
    elif isinstance(obj, Box2D):
        xmin = obj.xmin * scale_x
        ymin = obj.ymin * scale_y
        xmax = obj.xmax * scale_x
        ymax = obj.ymax * scale_y
        if return_integers:
            return Box2D(int(xmin), int(ymin), int(xmax), int(ymax), obj.description)
        return Box2D(xmin, ymin, xmax, ymax, obj.description)
    else:
        raise TypeError("Unsupported type")


normalize_point = partial(scale, dst_width=1, dst_height=1, return_integers=False)
denormalize_point = partial(scale, src_width=1, src_height=1, return_integers=False)
normalize_point_1k = partial(
    scale, dst_width=1000, dst_height=1000, return_integers=True
)
denormalize_point_1k = partial(
    scale, src_width=1000, src_height=1000, return_integers=False
)


def draw_box(image: Image.Image | str, box: Box2D):
    if isinstance(image, str):
        image = Image.open(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((box.xmin, box.ymin, box.xmax, box.ymax), outline="red", width=2)
    return image


def draw_point(image: Image.Image, point: Point):
    draw = ImageDraw.Draw(image)
    draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red")
    return image
