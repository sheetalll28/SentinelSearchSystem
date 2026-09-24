"""
Unified LLM interface. Every agent talks to `LLM.chat(...)` and never touches
a vendor SDK directly, so swapping providers is a one-line config change.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CONFIG
from cost_tracker import COST_TRACKER


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, max_tokens: int = 1500) -> str:
        ...


class ClaudeLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        key = api_key or CONFIG.anthropic_api_key
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env (and no api_key override provided)")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = CONFIG.anthropic_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def chat(self, system: str, user: str, max_tokens: int = 1500) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        COST_TRACKER.log(
            provider="anthropic",
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        key = api_key or CONFIG.openai_api_key
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in .env (and no api_key override provided)")
        self.client = OpenAI(api_key=key)
        self.model = CONFIG.openai_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def chat(self, system: str, user: str, max_tokens: int = 1500) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        COST_TRACKER.log(
            provider="openai",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        return resp.choices[0].message.content or ""


class GroqLLM(BaseLLM):
    """Groq offers a free-tier API (no credit card required) running Llama
    models with very fast inference. Its API is OpenAI-compatible, so this
    just points the openai SDK at Groq's endpoint instead of OpenAI's —
    useful when deploying somewhere Ollama can't run and Claude/OpenAI keys
    aren't available."""
    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        key = api_key or CONFIG.groq_api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in .env (and no api_key override provided)")
        self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        self.model = CONFIG.groq_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def chat(self, system: str, user: str, max_tokens: int = 1500) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        COST_TRACKER.log(
            provider="groq",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        return resp.choices[0].message.content or ""


class OllamaLLM(BaseLLM):
    def __init__(self):
        import ollama
        self._ollama = ollama
        self.client = ollama.Client(host=CONFIG.ollama_host)
        self.model = CONFIG.ollama_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def chat(self, system: str, user: str, max_tokens: int = 1500) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"num_predict": max_tokens},
        )
        # Ollama is local / free — still log for consistent reporting (cost = 0).
        COST_TRACKER.log(
            provider="ollama",
            model=self.model,
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
        )
        return resp["message"]["content"]


def get_llm(provider: Optional[str] = None, api_key: Optional[str] = None) -> BaseLLM:
    """Builds an LLM client for the given provider. `provider`/`api_key`, when
    passed explicitly, override the values loaded from .env at startup —
    this is what lets the CLI's --provider flag and the web UI's per-run
    provider switch actually take effect, since CONFIG's fields are fixed
    once at import time and mutating os.environ afterward has no effect on it."""
    provider = (provider or CONFIG.llm_provider).lower()
    if provider == "anthropic":
        return ClaudeLLM(api_key=api_key)
    if provider == "openai":
        return OpenAILLM(api_key=api_key)
    if provider == "groq":
        return GroqLLM(api_key=api_key)
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use anthropic | openai | groq | ollama.")
