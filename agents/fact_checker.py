from __future__ import annotations
from typing import Dict, List
from agents.base_agent import BaseAgent
from rag.vector_store import VectorStore


class FactChecker(BaseAgent):
    name = "Agent C — Fact Checker"
    persona = (
        "You are the Fact Checker in a multi-agent research system. You are given a set "
        "of claims from a web scout, plus optional supporting context retrieved from a "
        "local knowledge base (RAG). For each claim, assess it as SUPPORTED, CONTRADICTED, "
        "or UNVERIFIED based on the provided context and your own knowledge caution. "
        "Respond ONLY with a JSON array of objects: "
        '{"claim": str, "verdict": "SUPPORTED"|"CONTRADICTED"|"UNVERIFIED", "note": str}.'
    )

    def __init__(self, llm=None, vector_store: VectorStore | None = None):
        super().__init__(llm)
        self.store = vector_store or VectorStore()

    def verify(self, scout_result: Dict) -> Dict:
        claims: List[str] = scout_result.get("claims", [])
        if not claims:
            return {**scout_result, "verified_claims": []}

        rag_hits = self.store.query(scout_result["sub_task"], top_k=4)
        rag_context = "\n".join(f"- ({h['source']}) {h['text'][:300]}" for h in rag_hits) or "(no local knowledge base matches)"

        claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        raw = self.think(
            f"Sub-task: {scout_result['sub_task']}\n\n"
            f"Claims to verify:\n{claims_block}\n\n"
            f"Local knowledge base context:\n{rag_context}\n\n"
            "Return the JSON verdict array now."
        )
        verified = self.parse_json(raw)
        if not isinstance(verified, list):
            verified = [{"claim": c, "verdict": "UNVERIFIED", "note": "parse failure"} for c in claims]

        return {**scout_result, "verified_claims": verified, "rag_sources_used": [h["source"] for h in rag_hits]}
