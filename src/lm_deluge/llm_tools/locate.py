# # utilities for locating things in images
# from dataclasses import dataclass
# from typing import Literal

# from lm_deluge.util.json import load_json
# from lm_deluge.util.spatial import Box2D, Point


# @dataclass
# class LocatePrompt:
#     description: str
#     task: Literal["click", "detect"] = "click"
#     orientation: Literal["xy", "yx"] = "xy"
#     origin: Literal["top-left", "bottom-left"] = "top-left"
#     coords: Literal["absolute", "relative-1", "relative-1k"] = "absolute"
#     height: int | None = None
#     width: int | None = None
#     output_type: Literal["point", "points", "box", "boxes"] = "point"
#     output_fmt: Literal["json", "xml"] = "xml"

#     def to_string(self) -> str:
#         """Compose the full prompt string based on the current configuration."""
#         parts: list[str] = []

#         if self.task == "click":
#             parts.append(
#                 "Given an instruction, determine where to click in the image to complete it. "
#             )
#             parts.append(f"\n\nINSTRUCTION: {self.description}\n\n")
#         else:
#             if self.output_type.endswith("s"):
#                 parts.append(
#                     "Given a description of an object to locate, find ALL instances of that object in the image."
#                 )
#             else:
#                 parts.append(
#                     "Given a description of an object to locate, find that object in the image."
#                 )
#             parts.append(f"\n\nDESCRIPTION: {self.description}\n\n")

#         if self.output_type == "point":
#             point_type = "(x, y)" if self.orientation == "xy" else "(y, x)"
#             parts.append(f"Your response should be a single {point_type} point")
#             if self.output_fmt == "xml":
#                 parts.append("enclosed in <point></point> tags.")
#             else:
#                 parts.append('formatted as JSON, like {"point": [0, 1]}.')

#         elif self.output_type == "points":
#             point_type = "(x, y)" if self.orientation == "xy" else "(y, x)"
#             parts.append(f"Your response should be a series of {point_type} points,")
#             if self.output_fmt == "xml":
#                 parts.append("each one enclosed in <point></point> tags.")
#             else:
#                 parts.append(
#                     'formatted as a JSON array, like [{"point": [0, 1]}, {"point": [1, 0]}].'
#                 )

#         elif self.output_type == "box":
#             box_type = (
#                 "(x0, y0, x1, y1)" if self.orientation == "xy" else "(y0, x0, y1, x1)"
#             )
#             parts.append(f"Your response should be a {box_type} bounding box,")

#             if self.output_fmt == "xml":
#                 parts.append("enclosed in <box></box> tags.")
#             else:
#                 parts.append('formatted as JSON, like {"box_2d": [0, 0, 1, 1]}')

#         elif self.output_type == "boxes":
#             box_type = (
#                 "(x0, y0, x1, y1)" if self.orientation == "xy" else "(y0, x0, y1, x1)"
#             )
#             parts.append(
#                 f"Your response should be a series of {box_type} bounding boxes,"
#             )
#             if self.output_fmt == "xml":
#                 parts.append("each one enclosed in <box></box> tags.")
#             else:
#                 parts.append(
#                     'formatted as a JSON array, like [{"box_2d": [0, 0, 1, 1]}, {"box_2d": [0.5, 0.5, 1, 1]}].'
#                 )

#         if self.coords == "absolute":
#             parts.append(
#                 "The returned coordinates should be absolute pixel coordinates in the image. "
#                 f"The image has a height of {self.height} pixels and a width of {self.width} pixels. "
#             )
#             if self.origin == "top-left":
#                 parts.append("The origin (0, 0) is at the top-left of the image.")
#             else:
#                 parts.append("The origin (0, 0) is at the bottom-left of the image.")
#         elif self.coords == "relative-1":
#             parts.append(
#                 "The returned coordinates should be relative coordinates where x are between 0 and 1. "
#             )
#             if self.origin == "top-left":
#                 parts.append(
#                     "The origin (0, 0) is at the top-left of the image, and (1, 1) is at the bottom-right."
#                 )
#             else:
#                 parts.append(
#                     "The origin (0, 0) is at the bottom-left of the image, and (1, 1) is at the top-right."
#                 )
#         elif self.coords == "relative-1k":
#             parts.append(
#                 "The returned coordinates should be relative coordinates where x are between 0 and 1000. "
#             )
#             if self.origin == "top-left":
#                 parts.append(
#                     "The origin (0, 0) is at the top-left of the image, and (1000, 1000) is at the bottom-right."
#                 )
#             else:
#                 parts.append(
#                     "The origin (0, 0) is at the bottom-left of the image, and (1000, 1000) is at the top-right."
#                 )

#         parts.append(
#             "Return JUST the structured output, no prelude or commentary needed."
#         )

#         result = ""
#         for part in parts:
#             if part.startswith("\n") or result.endswith("\n"):
#                 result += part
#             else:
#                 result += " " + part

#         return result.strip()

#     def parse_output(self, output: str) -> Point | Box2D | list[Point] | list[Box2D]:
#         if self.output_fmt == "json":
#             loaded = load_json(output)
#             if self.output_type == "point":
#                 assert isinstance(loaded, dict)
#                 if self.orientation == "xy":
#                     x, y = loaded["point"]
#                 else:
#                     y, x = loaded["point"]

#                     return Point(x=)

#         else:
#             pass

#         return []


# def locate_point():
#     pass


# def locate_points():
#     pass


# def locate_box():
#     pass


# def locate_boxes():
#     pass
