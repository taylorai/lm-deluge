from .text import Text
from .image import Image
from .file import File


def json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, Text):
        return {"type": "text", "text": value.text}
    if isinstance(value, Image):
        w, h = value.size
        return {"type": "image", "tag": f"<Image ({w}×{h})>"}
    if isinstance(value, File):
        size = value.size
        return {"type": "file", "tag": f"<File ({size} bytes)>"}
    return repr(value)
