"""Thin Anthropic API client with retry logic."""
from __future__ import annotations

import logging
import time

import anthropic

from topic_modeling.config.schema import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Wraps the Anthropic Messages API with retry and structured output helpers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str) -> str:
        """Send a single user prompt and return the text response."""
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                message = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"LLM call failed (attempt {attempt}/{self.config.max_retries}): {exc}"
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds * attempt)

        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts"
        ) from last_exc
