"""Provider-agnostic LLM client with retry logic."""
from __future__ import annotations

import logging
import os
import threading
import time

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
    """Provider-agnostic LLM client with retry and rate limiting."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._provider = config.provider
        self._client = self._build_client()
        self._limiter = (
            _RateLimiter(config.tokens_per_minute)
            if config.tokens_per_minute is not None
            else None
        )

    def _build_client(self):
        if self._provider == "anthropic":
            import anthropic  # type: ignore

            kwargs = {}
            api_key = _resolve_api_key(
                explicit_env_var=self.config.api_key_env,
                default_env_var="ANTHROPIC_API_KEY",
            )
            if api_key:
                kwargs["api_key"] = api_key
            return anthropic.Anthropic(**kwargs)

        if self._provider in ("openai", "grok"):
            from openai import OpenAI  # type: ignore

            default_env = "OPENAI_API_KEY" if self._provider == "openai" else "XAI_API_KEY"
            api_key = _resolve_api_key(
                explicit_env_var=self.config.api_key_env,
                default_env_var=default_env,
            )
            base_url = self.config.api_base
            if self._provider == "grok" and not base_url:
                base_url = "https://api.x.ai/v1"
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs)

        if self._provider == "gemini":
            import google.generativeai as genai  # type: ignore

            api_key = _resolve_api_key(
                explicit_env_var=self.config.api_key_env,
                default_env_var="GOOGLE_API_KEY",
            )
            if api_key:
                genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.config.model)

        raise ValueError(f"Unsupported LLM provider: {self._provider!r}")

    def complete(self, prompt: str) -> str:
        """Send a single user prompt and return the text response."""
        if self._limiter is not None:
            estimated_tokens = int(len(prompt.split()) * 1.3)
            self._limiter.acquire(estimated_tokens)

        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if self._provider == "anthropic":
                    message = self._client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return message.content[0].text

                if self._provider in ("openai", "grok"):
                    response = self._client.chat.completions.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content or ""

                if self._provider == "gemini":
                    response = self._client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.config.temperature,
                            "max_output_tokens": self.config.max_tokens,
                        },
                    )
                    return getattr(response, "text", "") or ""

                raise ValueError(f"Unsupported LLM provider: {self._provider!r}")
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


def _resolve_api_key(explicit_env_var: str | None, default_env_var: str) -> str | None:
    env_var = explicit_env_var or default_env_var
    return os.getenv(env_var)
