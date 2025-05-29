from pydantic import BaseModel
from typing import Literal


class SamplingParams(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 512
    reasoning_effort: Literal["low", "medium", "high", None] = None
    logprobs: bool = False
    top_logprobs: int | None = None

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


class ComputerUseParams(BaseModel):
    enabled: bool = False
    display_width: int = 1024
    display_height: int = 768
