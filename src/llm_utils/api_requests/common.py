# from .vertex import VertexAnthropicRequest, GeminiRequest
# from .bedrock import BedrockAnthropicRequest, MistralBedrockRequest
# from .deepseek import DeepseekRequest
from .openai import OpenAIRequest
from .cohere import CohereRequest
from .anthropic import AnthropicRequest

CLASSES = {
    "openai": OpenAIRequest,
    # "deepseek": DeepseekRequest,
    "anthropic": AnthropicRequest,
    # "vertex_anthropic": VertexAnthropicRequest,
    # "vertex_gemini": GeminiRequest,
    "cohere": CohereRequest,
    # "bedrock_anthropic": BedrockAnthropicRequest,
    # "bedrock_mistral": MistralBedrockRequest,
    # "mistral": MistralRequest,
}
