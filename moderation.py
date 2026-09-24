"""
Lightweight moderation gate for both input queries and generated output.
Uses a local keyword/heuristic check by default (no external dependency),
with an optional hook to plug in OpenAI's Moderation API if OPENAI_API_KEY is set.
"""
from __future__ import annotations
from config import CONFIG

_BLOCK_TERMS = [
    "bomb making", "child sexual", "credit card dump", "how to make a weapon",
    "synthesize ricin", "kill instructions",
]


def _heuristic_flag(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _BLOCK_TERMS)


def moderate(text: str) -> tuple[bool, str]:
    """Returns (is_flagged, reason)."""
    if not CONFIG.enable_moderation:
        return False, ""

    if CONFIG.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=CONFIG.openai_api_key)
            result = client.moderations.create(input=text)
            flagged = result.results[0].flagged
            if flagged:
                cats = [c for c, v in result.results[0].categories.__dict__.items() if v]
                return True, f"OpenAI moderation flagged categories: {cats}"
            return False, ""
        except Exception:
            pass  # fall through to heuristic if API call fails

    if _heuristic_flag(text):
        return True, "Matched local heuristic block-list."
    return False, ""
