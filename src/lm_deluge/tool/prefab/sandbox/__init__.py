"""
Sandbox tools for lm-deluge.

This module uses lazy imports to avoid requiring all dependencies at once.
Import specific sandbox types as needed:
    from lm_deluge.tool.prefab.sandbox import ModalSandbox
    from lm_deluge.tool.prefab.sandbox import DockerSandbox
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .daytona_sandbox import DaytonaSandbox as DaytonaSandbox
    from .docker_sandbox import DockerSandbox as DockerSandbox
    from .fargate_sandbox import FargateSandbox as FargateSandbox
    from .modal_sandbox import ModalSandbox as ModalSandbox
    from .pybubble_sandbox import PybubbleSandbox as PybubbleSandbox
    from .seatbelt_sandbox import SandboxMode as SandboxMode
    from .seatbelt_sandbox import SeatbeltSandbox as SeatbeltSandbox

__all__ = [
    "DaytonaSandbox",
    "DockerSandbox",
    "FargateSandbox",
    "ModalSandbox",
]

# PybubbleSandbox is Linux only
if sys.platform == "linux":
    __all__.append("PybubbleSandbox")

# SeatbeltSandbox is macOS only
if sys.platform == "darwin":
    __all__.extend(["SandboxMode", "SeatbeltSandbox"])

# Mapping of names to their module paths for lazy loading
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DaytonaSandbox": (".daytona_sandbox", "DaytonaSandbox"),
    "DockerSandbox": (".docker_sandbox", "DockerSandbox"),
    "FargateSandbox": (".fargate_sandbox", "FargateSandbox"),
    "ModalSandbox": (".modal_sandbox", "ModalSandbox"),
    "PybubbleSandbox": (".pybubble_sandbox", "PybubbleSandbox"),
    "SandboxMode": (".seatbelt_sandbox", "SandboxMode"),
    "SeatbeltSandbox": (".seatbelt_sandbox", "SeatbeltSandbox"),
}


def __getattr__(name: str):
    """Lazy import handler for sandbox tools."""
    if name in _LAZY_IMPORTS:
        # PybubbleSandbox only on Linux
        if name == "PybubbleSandbox" and sys.platform != "linux":
            raise AttributeError(
                f"{name} is only available on Linux (current platform: {sys.platform})"
            )

        # SeatbeltSandbox only on macOS
        if name in ("SandboxMode", "SeatbeltSandbox") and sys.platform != "darwin":
            raise AttributeError(
                f"{name} is only available on macOS (current platform: {sys.platform})"
            )

        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
