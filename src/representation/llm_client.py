"""
LLM Client — Dual-provider interface (Ollama + OpenAI).

Provides a synchronous LLM call interface with:
- TUM Ollama as primary provider (via OLLAMA_HOST env var)
- OpenAI API as fallback (via OPENAI_API_KEY env var)
- File-based response cache for reproducibility and cost savings
- Mock mode for CI/testing (no live LLM calls)
- Automatic failover with backoff

Configuration is driven entirely by experiment YAML (representation_config).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Synchronous LLM client with dual-provider failover and caching.

    Args:
        config: Representation config dict from experiment YAML. Relevant keys:
            - ``llm_provider``: ``"ollama"`` (default), ``"openai"``, or ``"auto"``
            - ``llm_model``: Model name (provider-specific)
            - ``llm_temperature``: Sampling temperature (default 0.0)
            - ``llm_mock``: If ``True``, return mock responses (for CI)
            - ``llm_cache_dir``: Cache directory (default ``data/llm_cache``)
            - ``ollama_host``: Ollama server URL (default from ``OLLAMA_HOST`` env)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.mock_mode: bool = self.config.get("llm_mock", False)
        self.temperature: float = self.config.get("llm_temperature", 0.0)
        self.model: str = self.config.get("llm_model", "qwen3.5:122b")
        self.provider: str = self.config.get("llm_provider", "auto")

        # Cache
        cache_dir = self.config.get("llm_cache_dir", "data/llm_cache")
        self.cache_dir = Path(cache_dir) / self.model.replace("/", "_").replace(":", "_")
        if not self.mock_mode:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Provider state
        self.ollama_host: str = self.config.get(
            "ollama_host", os.getenv("OLLAMA_HOST", "https://ollama.sps.ed.tum.de")
        )
        self._ollama_available: Optional[bool] = None
        self._openai_available: Optional[bool] = None
        self._ollama_backoff_until: float = 0.0

        # Metrics
        self._total_calls: int = 0
        self._cache_hits: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_latency_s: float = 0.0
        self._last_latency_s: float = 0.0
        self._last_provider: str = "none"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User-level prompt with the actual query.
            temperature: Override default temperature for this call.
            json_mode: Request JSON output format (provider-specific).

        Returns:
            The LLM's text response.

        Raises:
            RuntimeError: If no LLM provider is available and not in mock mode.
        """
        if self.mock_mode:
            return self._mock_response(user_prompt)

        # Check cache
        cache_key = self._cache_key(system_prompt, user_prompt, temperature)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            self._total_calls += 1
            self._last_provider = "cache"
            self._last_latency_s = 0.0
            return cached

        temp = temperature if temperature is not None else self.temperature
        response = self._call_with_failover(system_prompt, user_prompt, temp, json_mode)

        logger.debug(
            "LLM [%s] prompt: %.200s...", self._last_provider, user_prompt[:200]
        )
        logger.debug(
            "LLM [%s] response: %.500s", self._last_provider, response[:500]
        )

        # Cache the response (with prompts for debugging)
        self._cache_put(cache_key, response, system_prompt, user_prompt)
        return response

    def get_metrics(self) -> Dict[str, float]:
        """Return LLM client metrics."""
        return {
            "llm_api_calls": float(self._total_calls),
            "llm_cache_hits": float(self._cache_hits),
            "llm_cache_hit_rate": (
                self._cache_hits / self._total_calls if self._total_calls > 0 else 0.0
            ),
            "llm_total_latency_s": self._total_latency_s,
            "llm_last_latency_s": self._last_latency_s,
            "llm_tokens_prompt": float(self._total_prompt_tokens),
            "llm_tokens_completion": float(self._total_completion_tokens),
        }

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------

    def _call_with_failover(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Try Ollama first, fall back to OpenAI."""
        self._total_calls += 1
        now = time.time()

        providers = self._resolve_provider_order()
        last_error: Exception | None = None

        max_retries = self.config.get("llm_retries", 8)
        backoff_cap_s = self.config.get("llm_backoff_cap_s", 60)
        ollama_cooldown_s = self.config.get("llm_ollama_cooldown_s", 300)
        for provider in providers:
            for attempt in range(max_retries + 1):
                try:
                    t0 = time.perf_counter()
                    if provider == "ollama":
                        response = self._call_ollama(system_prompt, user_prompt, temperature, json_mode)
                    else:
                        response = self._call_openai(system_prompt, user_prompt, temperature, json_mode)
                    elapsed = time.perf_counter() - t0

                    self._last_latency_s = elapsed
                    self._total_latency_s += elapsed
                    self._last_provider = provider
                    return response

                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = min(15 * (2 ** attempt), backoff_cap_s)
                        logger.warning(
                            "LLM provider '%s' attempt %d/%d failed: %s — retrying in %ds",
                            provider, attempt + 1, max_retries + 1, e, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning("LLM provider '%s' failed after %d attempts: %s", provider, max_retries + 1, e)
                        if provider == "ollama":
                            self._ollama_backoff_until = now + ollama_cooldown_s

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    def _resolve_provider_order(self) -> List[str]:
        """Determine provider order based on config and availability."""
        now = time.time()

        if self.provider == "ollama":
            return ["ollama"]
        elif self.provider == "openai":
            return ["openai"]

        # auto: prefer Ollama, fall back to OpenAI
        order = []
        if now >= self._ollama_backoff_until:
            order.append("ollama")
        if os.getenv("OPENAI_API_KEY"):
            order.append("openai")
        if not order and self._ollama_backoff_until > now:
            # Backoff expired check
            order.append("ollama")
        if not order:
            order.append("ollama")  # Will fail with clear error
        return order

    def _call_ollama(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Call Ollama API.

        Uses HTTP streaming (``stream: true``) by default so the gateway in
        front of the TUM Ollama VM sees continuous activity and does not
        return 504. The final assembled content is returned to callers
        exactly as if the call had been non-streaming. Set
        ``llm_stream: false`` in the representation config to fall back to
        the legacy non-streaming path.
        """
        import requests

        url = f"{self.ollama_host}/api/chat"
        stream = bool(self.config.get("llm_stream", True))
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        # Separate connect (fast fail if server unreachable) vs read (slow for large models)
        connect_timeout = self.config.get("llm_connect_timeout", 15)
        read_timeout = self.config.get("llm_timeout", 600)

        if not stream:
            resp = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
            resp.raise_for_status()
            data = resp.json()
            if "eval_count" in data:
                self._total_completion_tokens += data["eval_count"]
            if "prompt_eval_count" in data:
                self._total_prompt_tokens += data["prompt_eval_count"]
            return data["message"]["content"]

        # Streaming path: accumulate message.content chunks until done.
        content_parts: List[str] = []
        eval_count = 0
        prompt_eval_count = 0
        with requests.post(
            url, json=payload, timeout=(connect_timeout, read_timeout), stream=True
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                message = chunk.get("message") or {}
                content_parts.append(message.get("content", ""))
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    break

        if eval_count:
            self._total_completion_tokens += eval_count
        if prompt_eval_count:
            self._total_prompt_tokens += prompt_eval_count
        return "".join(content_parts)

    def _call_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI()  # Uses OPENAI_API_KEY env var
        model = self.config.get("openai_model", "gpt-4o-mini")

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Track tokens
        if response.usage:
            self._total_prompt_tokens += response.usage.prompt_tokens
            self._total_completion_tokens += response.usage.completion_tokens

        return choice.message.content or ""

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _mock_response(self, user_prompt: str) -> str:
        """Return a deterministic mock response for testing."""
        self._total_calls += 1
        self._last_provider = "mock"
        self._last_latency_s = 0.001
        # Return a valid JSON mode selection that tests can parse
        return json.dumps({
            "mode": "charging",
            "rationale": "Mock LLM response: defaulting to charging for safety.",
        })

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, system_prompt: str, user_prompt: str, temperature: float | None) -> str:
        """Generate a deterministic cache key."""
        content = f"{self.model}|{temperature or self.temperature}|{system_prompt}|{user_prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        """Read a cached response."""
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("response")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _cache_put(
        self,
        key: str,
        response: str,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> None:
        """Write a response to cache (with prompts for debugging)."""
        path = self.cache_dir / f"{key}.json"
        data = {
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
