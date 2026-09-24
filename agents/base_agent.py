from __future__ import annotations
import json
import re
from typing import Any, Optional

from llm_providers import get_llm


class BaseAgent:
    """Shared plumbing for every SentinelSearch agent."""

    name: str = "BaseAgent"
    persona: str = "You are a helpful assistant."

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def think(self, user_prompt: str, max_tokens: int = 1500) -> str:
        return self.llm.chat(system=self.persona, user=user_prompt, max_tokens=max_tokens)

    @staticmethod
    def parse_json(raw: str) -> Optional[Any]:
        """Best-effort JSON extraction — LLMs often wrap JSON in prose or fences."""
        raw = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
        candidate = fence.group(1) if fence else raw
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
        return None
