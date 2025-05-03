"""
Models
"""

from abc import ABC, abstractmethod

import tiktoken
from pydantic import BaseModel
from vertexai.preview import tokenization


class PricingInfo(BaseModel):
    """Model pricing information."""

    prompt_cost_realtime: float
    """Cost per 1k tokens for real-time requests."""

    completion_cost_realtime: float
    """Cost per 1k tokens for real-time requests."""

    prompt_cost_batch: float
    """Cost per 1k tokens for batch requests."""

    completion_cost_batch: float
    """Cost per 1k tokens for batch requests."""

    _per: int = 1000

    def get_cost(self, input_tokens: int, output_tokens: int) -> dict:
        """Calculate costs based on input and output tokens."""
        input_rt_cost = input_tokens / self._per * self.prompt_cost_realtime
        input_batch_cost = input_tokens / self._per * self.prompt_cost_batch
        output_rt_cost = output_tokens / self._per * self.completion_cost_realtime
        output_batch_cost = output_tokens / self._per * self.completion_cost_batch
        return {
            "input": {
                "tokens": input_tokens,
                "rt_cost": input_rt_cost,
                "batch_cost": input_batch_cost,
            },
            "output": {
                "tokens": output_tokens,
                "rt_cost": output_rt_cost,
                "batch_cost": output_batch_cost,
            },
        }


class Model(BaseModel, ABC):
    """Base model class."""

    model_id: str
    """Model identifier."""

    name: str
    """Model name."""

    pricing: PricingInfo
    """Model pricing information."""

    tokenizer_model: str
    """Tokenizer model to use."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""

    def estimates_for_text(self, text_in: str, text_out: str) -> dict:
        """Estimate token counts and costs for the given input and output text."""
        input_tokens = self.count_tokens(text_in)
        output_tokens = self.count_tokens(text_out)
        info = self.pricing.get_cost(input_tokens, output_tokens)
        return info


class OpenAIModel(Model):
    """OpenAI model class."""

    def count_tokens(self, text) -> int:
        """Count tokens in text using OpenAI's tiktoken library."""
        encoding = tiktoken.get_encoding(self.tokenizer_model)
        tokens = encoding.encode(text)
        return len(tokens)


class GoogleGeminiModel(Model):
    """Google Gemini model class."""

    def count_tokens(self, text) -> int:
        """Count tokens in text using Google's tokenizer."""
        tokenizer = tokenization.get_tokenizer_for_model(self.tokenizer_model)
        result = tokenizer.count_tokens(text)
        return result.total_tokens
