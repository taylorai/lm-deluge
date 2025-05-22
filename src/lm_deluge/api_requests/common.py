from .openai import OpenAIRequest
from .anthropic import AnthropicRequest
from .mistral import MistralRequest

CLASSES = {
    "openai": OpenAIRequest,
    "anthropic": AnthropicRequest,
    "mistral": MistralRequest,
}
