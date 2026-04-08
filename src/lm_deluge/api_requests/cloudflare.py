import os

from .openai import OpenAIRequest, _build_oa_chat_request


class CloudflareRequest(OpenAIRequest):
    """OpenAI-compatible handler for Cloudflare Workers AI.

    The only difference from vanilla OpenAI is that the api_base URL contains
    a ``{account_id}`` placeholder which must be resolved from the
    ``CLOUDFLARE_ACCOUNT_ID`` environment variable at request time, and
    Cloudflare uses ``max_tokens`` instead of ``max_completion_tokens``.
    """

    async def build_request(self):
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is required "
                "for Cloudflare Workers AI models."
            )

        base = self.model.api_base.replace("{account_id}", account_id)
        self.url = f"{base}/chat/completions"

        base_headers = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic"]
        )

        self.request_json = await _build_oa_chat_request(self.model, self.context)

        # Cloudflare uses max_tokens, not max_completion_tokens
        if "max_completion_tokens" in self.request_json:
            self.request_json["max_tokens"] = self.request_json.pop(
                "max_completion_tokens"
            )

        # Cloudflare doesn't support reasoning_effort
        self.request_json.pop("reasoning_effort", None)
