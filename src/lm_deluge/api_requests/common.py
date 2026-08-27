from .anthropic import AnthropicRequest
from .bedrock import BedrockRequest
from .bedrock_nova import BedrockNovaRequest
from .cloudflare import CloudflareMoondreamRequest, CloudflareRequest
from .gemini import GeminiRequest
from .mistral import MistralRequest
from .moondream import MoondreamRequest
from .nvidia import NVIDIARequest
from .openai import OpenAIRequest, OpenAIResponsesRequest

CLASSES = {
    "openai": OpenAIRequest,
    "openai-responses": OpenAIResponsesRequest,
    "anthropic": AnthropicRequest,
    "cloudflare": CloudflareRequest,
    "cloudflare-moondream": CloudflareMoondreamRequest,
    "mistral": MistralRequest,
    "bedrock": BedrockRequest,
    "bedrock-nova": BedrockNovaRequest,
    "gemini": GeminiRequest,
    "nvidia": NVIDIARequest,
    "moondream": MoondreamRequest,
}
