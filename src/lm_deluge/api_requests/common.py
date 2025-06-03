from .openai import OpenAIRequest, OpenAIResponsesRequest
from .anthropic import AnthropicRequest
from .mistral import MistralRequest
from .bedrock import BedrockRequest
from .gemini import GeminiRequest

CLASSES = {
    "openai": OpenAIRequest,
    "openai-responses": OpenAIResponsesRequest,
    "anthropic": AnthropicRequest,
    "mistral": MistralRequest,
    "bedrock": BedrockRequest,
    "gemini": GeminiRequest,
}
