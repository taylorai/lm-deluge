import os

from .openai import OpenAIRequest, _build_oa_chat_request


class NVIDIARequest(OpenAIRequest):
    """OpenAI-compatible handler for NVIDIA's hosted NIM catalog."""

    async def build_request(self):
        self.url = f"{self.model.api_base}/chat/completions"
        base_headers = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic"]
        )

        self.request_json = await _build_oa_chat_request(self.model, self.context)

        # NVIDIA's hosted chat endpoint expects max_tokens.
        if "max_completion_tokens" in self.request_json:
            max_completion_tokens = self.request_json.pop("max_completion_tokens")
            self.request_json.setdefault("max_tokens", max_completion_tokens)
