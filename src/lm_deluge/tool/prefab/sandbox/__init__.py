import sys

from .daytona_sandbox import DaytonaSandbox
from .docker_sandbox import DockerSandbox
from .fargate_sandbox import FargateSandbox
from .modal_sandbox import ModalSandbox

__all__ = [
    "DaytonaSandbox",
    "DockerSandbox",
    "FargateSandbox",
    "ModalSandbox",
]

# SeatbeltSandbox is macOS only
if sys.platform == "darwin":
    from .seatbelt_sandbox import SandboxMode, SeatbeltSandbox  # noqa

    __all__.extend(["SandboxMode", "SeatbeltSandbox"])
