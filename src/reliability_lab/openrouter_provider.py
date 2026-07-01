"""Real LLM provider backed by OpenRouter.

This is an *optional* extension to the lab: the graded tests all run against
``FakeLLMProvider`` (no network, no keys). ``OpenRouterProvider`` implements the
exact same interface — ``.name``, ``.fail_rate``, ``.complete(prompt) ->
ProviderResponse`` and raises ``ProviderError`` on failure — so it drops straight
into ``ReliabilityGateway`` and every reliability mechanism (circuit breaker,
cache, fallback chain) works against a live model.

Credentials are read from the environment / ``.env``:

    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_MODEL=google/gemini-2.5-flash-lite

Only the Python standard library is used (``urllib``), so no extra dependency is
added to ``pyproject.toml``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from reliability_lab.providers import ProviderError, ProviderResponse

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


def load_env(path: str | Path = ".env") -> None:
    """Minimal ``.env`` loader (no python-dotenv dependency).

    Only sets variables that are not already present in the environment, so real
    environment variables always win over the file.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class OpenRouterProvider:
    """OpenRouter-backed provider, drop-in compatible with ``FakeLLMProvider``."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        api_key: str | None = None,
        cost_per_1k_tokens: float = 0.0,
        timeout_seconds: float = 30.0,
        max_tokens: int = 512,
    ) -> None:
        self.name = name
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set — add it to your .env or call load_env()."
            )
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        # Interface parity with FakeLLMProvider (used by build_gateway / breakers).
        self.fail_rate = 0.0
        self.base_latency_ms = 0

    def complete(self, prompt: str) -> ProviderResponse:
        """Call OpenRouter and return a ``ProviderResponse``.

        Any transport, HTTP, or payload error is normalised to ``ProviderError``
        so the circuit breaker and gateway fallback logic treat a real API
        failure exactly like a simulated one.
        """
        start = time.perf_counter()
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            OPENROUTER_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/day10-reliability-lab",
                "X-Title": "Day10 Reliability Lab",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore") if exc.fp else exc.reason
            raise ProviderError(f"{self.name} HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"{self.name} network error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} invalid JSON response: {exc}") from exc

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name} malformed response: {body}") from exc

        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or max(1, len(prompt.split())))
        output_tokens = int(usage.get("completion_tokens") or max(1, len(text.split())))
        cost = (input_tokens + output_tokens) / 1000.0 * self.cost_per_1k_tokens
        latency_ms = (time.perf_counter() - start) * 1000

        return ProviderResponse(
            provider=self.name,
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
        )
