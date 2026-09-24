from __future__ import annotations
from typing import List, Dict
from agents.base_agent import BaseAgent
from search_tools import web_search, fetch_page_text


class WebScout(BaseAgent):
    name = "Agent B — Web Scout"
    persona = (
        "You are the Web Scout in a multi-agent research system. Given raw search "
        "snippets for a sub-task, extract the key factual claims relevant to the sub-task. "
        "Respond ONLY with a JSON array of short factual claim strings (no opinions, no fluff)."
    )

    def execute(self, sub_task: str, max_results: int = 4, fetch_top: int = 1) -> Dict:
        results = web_search(sub_task, max_results=max_results)

        # Optionally fetch full text of the top result(s) for richer grounding
        enriched = []
        for i, r in enumerate(results):
            snippet = r["snippet"]
            if i < fetch_top and r["url"]:
                full = fetch_page_text(r["url"])
                if not full.startswith("[fetch failed") and not full.startswith("[fetch blocked"):
                    snippet = full[:2000]
            enriched.append({**r, "snippet": snippet})

        from security import sanitize_external_content

        sanitized_any = False
        clean_enriched = []
        for r in enriched:
            clean_snippet, flagged = sanitize_external_content(r["snippet"], source=r["url"] or r["title"])
            sanitized_any = sanitized_any or flagged
            clean_enriched.append({**r, "snippet": clean_snippet})

        combined_snippets = "\n\n".join(f"[{r['title']}]({r['url']}): {r['snippet']}" for r in clean_enriched)
        raw = self.think(
            f"Sub-task: {sub_task}\n\nRaw search results:\n{combined_snippets}\n\n"
            "Extract the key factual claims now, as a JSON array of strings."
        )
        claims = self.parse_json(raw)
        if not isinstance(claims, list):
            claims = [raw.strip()] if raw.strip() else []

        return {
            "sub_task": sub_task,
            "sources": [{"title": r["title"], "url": r["url"]} for r in enriched],
            "claims": claims,
            "security_flag": sanitized_any,
        }
