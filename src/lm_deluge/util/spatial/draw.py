from pathlib import Path

from PIL import Image, ImageDraw

from .core import Box2D, CoordinateSpace, Point, Polygon, SpatialResult


def draw_box(image: Image.Image | str | Path, box: Box2D):
    pil_image = _coerce_image(image)
    if box.space != CoordinateSpace.PIXELS:
        box = box.to_pixels(pil_image.width, pil_image.height)
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle((box.xmin, box.ymin, box.xmax, box.ymax), outline="red", width=2)
    return pil_image


def draw_point(image: Image.Image | str | Path, point: Point):
    pil_image = _coerce_image(image)
    if point.space != CoordinateSpace.PIXELS:
        point = point.to_pixels(pil_image.width, pil_image.height)
    draw = ImageDraw.Draw(pil_image)
    draw.ellipse((point.x - 2, point.y - 2, point.x + 2, point.y + 2), fill="red")
    return pil_image


def draw_polygon(image: Image.Image | str | Path, polygon: Polygon):
    pil_image = _coerce_image(image)
    if polygon.space != CoordinateSpace.PIXELS:
        polygon = polygon.to_pixels(pil_image.width, pil_image.height)
    draw = ImageDraw.Draw(pil_image)
    points = [(p.x, p.y) for p in polygon.points]
    if not points:
        return pil_image
    draw.line(points + [points[0]], fill="red", width=2)
    return pil_image


def draw_spatial(image: Image.Image | str | Path, result: SpatialResult):
    pil_image = _coerce_image(image)
    for box in result.boxes:
        draw_box(pil_image, box)
    for point in result.points:
        draw_point(pil_image, point)
    for polygon in result.polygons:
        draw_polygon(pil_image, polygon)
    return pil_image


def _coerce_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)
