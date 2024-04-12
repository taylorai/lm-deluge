from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 512