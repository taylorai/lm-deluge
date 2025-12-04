from typing import Any, List, Literal, TypedDict, Union

Coord = tuple[int, int]


class CUActionBase(TypedDict):
    kind: str | Any  # discriminator


class Click(CUActionBase):
    kind: Literal["click"]
    x: int | None  # if missing, current cursor position
    y: int | None
    button: Literal["left", "right", "middle", "back", "forward"]


class DoubleClick(CUActionBase):
    kind: Literal["double_click"]
    x: int | None  # if missing, current cursor position
    y: int | None


class Move(CUActionBase):
    kind: Literal["move"]
    x: int
    y: int


class Drag(CUActionBase):
    kind: Literal["drag"]
    start_x: int | None  # if missing, current cursor position
    start_y: int | None  # if missing, current cursor position
    path: List[Coord]  # path to drag after mousedown


class Scroll(CUActionBase):
    kind: Literal["scroll"]
    x: int | None  # if not provided, current cursor position
    y: int | None  # if not provided, current cursor position
    dx: int  # scroll_x in OpenAI
    dy: int  # scroll_y in OpenAI


class Keypress(CUActionBase):
    kind: Literal["keypress"]
    keys: List[str]


class Type(CUActionBase):
    kind: Literal["type"]
    text: str


class Wait(CUActionBase):
    kind: Literal["wait"]
    ms: int


class Screenshot(CUActionBase):
    kind: Literal["screenshot"]


class MouseDown(CUActionBase):
    kind: Literal["mouse_down"]
    button: Literal["left", "right", "middle", "back", "forward"]


class MouseUp(CUActionBase):
    kind: Literal["mouse_up"]
    button: Literal["left", "right", "middle", "back", "forward"]


class CursorPos(CUActionBase):
    kind: Literal["cursor_position"]


class HoldKey(CUActionBase):
    kind: Literal["hold_key"]
    key: str
    ms: int  # duration


class TripleClick(CUActionBase):
    kind: Literal["triple_click"]
    x: int | None  # if missing, current cursor position
    y: int | None


# ── Bash / Editor (provider‑independent) ────────────────────────────
class Bash(CUActionBase):
    kind: Literal["bash"]
    command: str | None
    restart: bool | None


class Edit(CUActionBase):
    kind: Literal["edit"]
    command: Literal["view", "create", "str_replace", "insert", "undo_edit"]
    path: str
    # optional, keep names identical to Anthropic spec
    file_text: str | None
    view_range: List[int] | None
    old_str: str | None
    new_str: str | None
    insert_line: int | None


CUAction = Union[
    Click,
    DoubleClick,
    TripleClick,
    MouseDown,
    MouseUp,
    Drag,
    Move,
    Scroll,
    Keypress,
    Type,
    HoldKey,
    Wait,
    Screenshot,
    CursorPos,
    Bash,
    Edit,
]
