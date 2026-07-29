"""
OpenAI-compatible chat transport for the OHCA classifier.

This module is intentionally small: it lets the classifier package talk to
vLLM, llama.cpp, oMLX, and other servers that expose /v1/chat/completions while
keeping the OHCA prompt and step logic independent of any one runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .config import QWEN_BASE_URL, QWEN_MODEL


RETRYABLE_STATUS = {409, 429, 500, 502, 503, 504}


@dataclass
class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client for deterministic scoring."""

    base_url: str = QWEN_BASE_URL
    model: str = QWEN_MODEL
    retries: int = 5
    backoff: float = 2.0
    timeout: int = 180
    send_seed: bool = True
    req_pause: float = 0.0
    disable_thinking: bool = True

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int | None,
        max_tokens: int,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.send_seed and seed is not None:
            payload["seed"] = seed
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {
                "enable_thinking": False,
                "preserve_thinking": False,
            }

        wait = self.backoff
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Endpoint {url} not reachable ({type(exc).__name__}: {exc})."
                    ) from exc
                time.sleep(wait)
                wait *= 2
                continue

            if response.status_code == 200:
                if self.req_pause:
                    time.sleep(self.req_pause)
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""

            body = (response.text or "")[:300]
            if response.status_code in RETRYABLE_STATUS and attempt < self.retries:
                extra = 3.0 if response.status_code == 409 else 0.0
                time.sleep(wait + extra)
                wait *= 2
                continue

            raise RuntimeError(
                f"Server returned HTTP {response.status_code} at {url} after "
                f"{attempt} attempt(s). Body: {body!r}."
            )

        raise RuntimeError(f"Endpoint {url} failed without a response.")
