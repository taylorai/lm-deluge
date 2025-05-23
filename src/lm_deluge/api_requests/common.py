from .openai import OpenAIRequest
from .anthropic import AnthropicRequest
from .mistral import MistralRequest
from .bedrock import BedrockRequest

CLASSES = {
    "openai": OpenAIRequest,
    "anthropic": AnthropicRequest,
    "mistral": MistralRequest,
    "bedrock": BedrockRequest,
}
