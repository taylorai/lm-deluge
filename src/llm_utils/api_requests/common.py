import asyncio
from tqdm import tqdm
from .anthropic import AnthropicRequest
from .vertex import VertexAnthropicRequest, GeminiRequest
from .bedrock import BedrockAnthropicRequest, MistralBedrockRequest
from .deepseek import DeepseekRequest
from .openai import OpenAIRequest
from .mistral import MistralRequest
from .cohere import CohereRequest
from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..models import APIModel

from typing import Optional, Callable

CLASSES = {
    "openai": OpenAIRequest,
    "deepseek": DeepseekRequest,
    "anthropic": AnthropicRequest,
    "vertex_anthropic": VertexAnthropicRequest,
    "vertex_gemini": GeminiRequest,
    "cohere": CohereRequest,
    "bedrock_anthropic": BedrockAnthropicRequest,
    "bedrock_mistral": MistralBedrockRequest,
    "mistral": MistralRequest,
}