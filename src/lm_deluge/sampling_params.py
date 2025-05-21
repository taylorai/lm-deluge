from dataclasses import dataclass
from typing import Literal


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 512
    reasoning_effort: Literal["low", "medium", "high", None] = None

    def to_vllm(self):
        try:
            from vllm import SamplingParams as VLLMSamplingParams  # pyright: ignore
        except ImportError as e:
            print(
                "Unable to import from vLLM. Make sure it's installed with `pip install vllm`."
            )
            raise e
        return VLLMSamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
