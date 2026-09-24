"""
Tracks token usage and estimated USD cost per query, per Phase 5 requirements.
Pricing is approximate (per 1M tokens) and easy to update as vendors change rates.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import List

PRICING_PER_MILLION = {
    # provider: (input $/1M, output $/1M) — update as needed
    "anthropic": (3.00, 15.00),
    "openai": (2.50, 10.00),
    "groq": (0.00, 0.00),  # free tier as of writing — check console.groq.com for current limits
    "ollama": (0.00, 0.00),  # local, free
}


@dataclass
class UsageEvent:
    ts: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    def __init__(self, log_path: str = "./data/usage_log.jsonl"):
        self.log_path = log_path
        self.events: List[UsageEvent] = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> UsageEvent:
        in_price, out_price = PRICING_PER_MILLION.get(provider, (0.0, 0.0))
        cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
        event = UsageEvent(time.time(), provider, model, input_tokens, output_tokens, cost)
        self.events.append(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event.__dict__) + "\n")
        return event

    def session_summary(self) -> dict:
        total_in = sum(e.input_tokens for e in self.events)
        total_out = sum(e.output_tokens for e in self.events)
        total_cost = sum(e.cost_usd for e in self.events)
        return {
            "calls": len(self.events),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "estimated_cost_usd": round(total_cost, 6),
        }


COST_TRACKER = CostTracker()
