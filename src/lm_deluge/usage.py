from dataclasses import dataclass


@dataclass
class Usage:
    """
    Unified usage tracking for all API providers.

    Tracks token usage including cache hits and writes for providers that support it.
    For providers that don't support caching, cache_read and cache_write will be None.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including both fresh input, cache writes, and cache reads."""
        result = self.input_tokens
        if self.cache_read_tokens is not None:
            result += self.cache_read_tokens
        if self.cache_write_tokens is not None:
            result += self.cache_write_tokens
        return result

    @property
    def total_tokens(self) -> int:
        """Total tokens processed (input + output)."""
        return self.total_input_tokens + self.output_tokens

    @property
    def has_cache_hit(self) -> bool:
        """Whether this request had any cache hits."""
        return self.cache_read_tokens is not None and self.cache_read_tokens > 0

    @property
    def has_cache_write(self) -> bool:
        """Whether this request wrote to cache."""
        return self.cache_write_tokens is not None and self.cache_write_tokens > 0

    @classmethod
    def from_anthropic_usage(cls, usage_data: dict) -> "Usage":
        """Create Usage from Anthropic API response usage data."""
        return cls(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
        )

    @classmethod
    def from_openai_usage(cls, usage_data: dict) -> "Usage":
        """Create Usage from OpenAI API response usage data.

        OpenAI supports prompt caching - cached tokens appear in prompt_tokens_details.cached_tokens.
        Caching is automatic for prompts over 1024 tokens.
        """
        prompt_tokens_details = usage_data.get("prompt_tokens_details", {})
        cached_tokens = (
            prompt_tokens_details.get("cached_tokens", 0)
            if prompt_tokens_details
            else 0
        )

        return cls(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            cache_read_tokens=cached_tokens if cached_tokens > 0 else 0,
            cache_write_tokens=0,  # OpenAI doesn't charge separately for cache writes
        )

    @classmethod
    def from_mistral_usage(cls, usage_data: dict) -> "Usage":
        """Create Usage from Mistral API response usage data."""
        return cls(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            cache_read_tokens=0,  # Mistral doesn't support caching
            cache_write_tokens=0,
        )

    @classmethod
    def from_gemini_usage(cls, usage_data: dict) -> "Usage":
        """Create Usage from Gemini API response usage data.

        Gemini supports context caching - cached tokens appear in cachedContentTokenCount.
        """
        cached_tokens = usage_data.get("cachedContentTokenCount", 0)

        return cls(
            input_tokens=usage_data.get("promptTokenCount", 0),
            output_tokens=usage_data.get("candidatesTokenCount", 0),
            cache_read_tokens=cached_tokens if cached_tokens > 0 else 0,
            cache_write_tokens=0,  # Gemini doesn't charge separately for cache writes
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_tokens": self.total_tokens,
            "has_cache_hit": self.has_cache_hit,
            "has_cache_write": self.has_cache_write,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        """Create Usage from dictionary."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
        )

    def __add__(self, other: "Usage") -> "Usage":
        """Add two Usage objects together."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=(
                (self.cache_read_tokens or 0) + (other.cache_read_tokens or 0)
            ),
            cache_write_tokens=(
                (self.cache_write_tokens or 0) + (other.cache_write_tokens or 0)
            ),
        )
