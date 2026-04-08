from .openai import OpenAIRequest, OpenAIResponsesRequest
from .anthropic import AnthropicRequest
from .cloudflare import CloudflareRequest
from .mistral import MistralRequest
from .bedrock import BedrockRequest
from .bedrock_nova import BedrockNovaRequest
from .gemini import GeminiRequest

CLASSES = {
    "openai": OpenAIRequest,
    "openai-responses": OpenAIResponsesRequest,
    "anthropic": AnthropicRequest,
    "cloudflare": CloudflareRequest,
    "mistral": MistralRequest,
    "bedrock": BedrockRequest,
    "bedrock-nova": BedrockNovaRequest,
    "gemini": GeminiRequest,
}
