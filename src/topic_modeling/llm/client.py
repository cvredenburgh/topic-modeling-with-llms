"""Thin Anthropic API client with retry logic."""
from __future__ import annotations

import logging
import threading
import time

import anthropic

from topic_modeling.config.schema import LLMConfig

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Token-per-minute window limiter (thread-safe)."""

    def __init__(self, tpm: int, safety_margin: float = 0.85):
        self._limit = int(tpm * safety_margin)
        self._used = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start
            if elapsed >= 60.0:
                self._used = 0
                self._window_start = now
                elapsed = 0.0

            if self._used + estimated_tokens > self._limit:
                wait = 60.0 - elapsed
                logger.debug(f"Rate limit: sleeping {wait:.1f}s (used={self._used}, limit={self._limit})")
                time.sleep(wait)
                self._used = 0
                self._window_start = time.monotonic()

            self._used += estimated_tokens


class LLMClient:
    """Wraps the Anthropic Messages API with retry and structured output helpers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = anthropic.Anthropic()
        self._limiter = (
            _RateLimiter(config.tokens_per_minute)
            if config.tokens_per_minute is not None
            else None
        )

    def complete(self, prompt: str) -> str:
        """Send a single user prompt and return the text response."""
        if self._limiter is not None:
            estimated_tokens = int(len(prompt.split()) * 1.3)
            self._limiter.acquire(estimated_tokens)

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
