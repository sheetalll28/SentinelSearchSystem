"""
Wires Agent A -> B -> C -> D together (Phase 4). Supports both:
  - sequential: sub-tasks run one at a time
  - hierarchical: sub-tasks run concurrently (B+C per sub-task in parallel threads)
before D synthesises the final report.
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Callable, Optional

from agents.lead_researcher import LeadResearcher
from agents.web_scout import WebScout
from agents.fact_checker import FactChecker
from agents.editor import Editor
from rag.vector_store import VectorStore
from moderation import moderate
from cost_tracker import COST_TRACKER
from llm_providers import get_llm


class SentinelSearchOrchestrator:
    def __init__(self, mode: str = "hierarchical", provider: Optional[str] = None,
                 api_key: Optional[str] = None, collection_name: Optional[str] = None):
        assert mode in ("sequential", "hierarchical")
        self.mode = mode
        # One explicit LLM client shared by all four agents — this is what
        # actually makes --provider / a UI provider switch take effect,
        # since CONFIG's fields are frozen at import time.
        llm = get_llm(provider=provider, api_key=api_key)
        store = VectorStore(collection_name=collection_name or "sentinel_search")
        self.lead = LeadResearcher(llm=llm)
        self.scout = WebScout(llm=llm)
        self.checker = FactChecker(llm=llm, vector_store=store)
        self.editor = Editor(llm=llm)

    def _run_subtask(self, sub_task: str) -> Dict:
        scout_result = self.scout.execute(sub_task)
        return self.checker.verify(scout_result)

    def run(self, query: str, status_cb: Optional[Callable[[str], None]] = None) -> Dict:
        def status(msg: str):
            if status_cb:
                status_cb(msg)
            else:
                print(msg)

        flagged, reason = moderate(query)
        if flagged:
            return {"error": f"Query blocked by moderation: {reason}", "stage": "input", "reason": reason}

        status(f"[{self.lead.name}] Deconstructing query...")
        sub_tasks: List[str] = self.lead.plan(query)
        status(f"  -> {len(sub_tasks)} sub-tasks: {sub_tasks}")

        verified_results: List[Dict] = []
        if self.mode == "sequential":
            for st in sub_tasks:
                status(f"[{self.scout.name}] Researching: {st}")
                res = self._run_subtask(st)
                status(f"[{self.checker.name}] Verified: {st}")
                verified_results.append(res)
        else:
            status(f"[{self.scout.name} + {self.checker.name}] Running {len(sub_tasks)} sub-tasks in parallel...")
            with ThreadPoolExecutor(max_workers=min(4, len(sub_tasks) or 1)) as pool:
                futures = {pool.submit(self._run_subtask, st): st for st in sub_tasks}
                for fut in as_completed(futures):
                    verified_results.append(fut.result())
                    status(f"  -> done: {futures[fut]}")

        status(f"[{self.editor.name}] Compiling final report...")
        report_md = self.editor.compile_report(query, verified_results)

        out_flagged, out_reason = moderate(report_md)
        if out_flagged:
            return {"error": f"Generated report blocked by moderation: {out_reason}", "stage": "output", "reason": out_reason}

        return {
            "query": query,
            "sub_tasks": sub_tasks,
            "results": verified_results,
            "report_markdown": report_md,
            "usage": COST_TRACKER.session_summary(),
        }
