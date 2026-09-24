from __future__ import annotations
from typing import List
from agents.base_agent import BaseAgent


class LeadResearcher(BaseAgent):
    name = "Agent A — Lead Researcher"
    persona = (
        "You are the Lead Researcher in a multi-agent research system. "
        "Your job is to deconstruct a user's research query into 3-5 concrete, "
        "non-overlapping, web-searchable sub-tasks that together cover the topic well. "
        "Respond ONLY with a JSON array of short sub-task strings. No prose, no markdown fences."
    )

    def plan(self, query: str) -> List[str]:
        raw = self.think(f"Research query: {query}\n\nReturn the JSON array of sub-tasks now.")
        parsed = self.parse_json(raw)
        if isinstance(parsed, list) and parsed:
            return [str(s) for s in parsed][:5]
        # graceful fallback if the model didn't return clean JSON
        lines = [l.strip("-• ").strip() for l in raw.splitlines() if l.strip()]
        return lines[:5] if lines else [query]
