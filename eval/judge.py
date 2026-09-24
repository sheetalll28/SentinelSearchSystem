"""
LLM-as-a-judge evaluation harness (Phase 5 requirement).

Runs SentinelSearch end-to-end on a set of sample queries, then uses a
second LLM call — a "judge" — to score each generated report on:
  - factual_accuracy   (1-5): are claims presented with appropriate
                        confidence, and are UNVERIFIED/CONTRADICTED claims
                        flagged rather than stated as fact?
  - completeness        (1-5): does the report actually address the query?
  - readability          (1-5): clear structure, no redundancy, good prose

The judge is deliberately a *separate* prompt/call from the Editor agent
that wrote the report, so it isn't just grading its own homework with the
same context window.

Usage:
    python -m eval.judge                     # run on the built-in sample set
    python -m eval.judge --n 3                # only run the first 3 samples
    python -m eval.judge --queries "q1" "q2"   # run on custom queries
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

from llm_providers import get_llm
from orchestrator import SentinelSearchOrchestrator

DEFAULT_SAMPLE_QUERIES = [
    "What is retrieval-augmented generation and why does it reduce hallucination?",
    "What are the main differences between supervised fine-tuning and RAG?",
    "What is the ReAct pattern for AI agents?",
    "What are common risks of prompt injection in AI agent systems?",
    "How does vector similarity search work in embedding-based retrieval?",
]

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring a research report produced by an
AI multi-agent system. You did not write this report — evaluate it independently and critically.

Score the report on three dimensions, each from 1 (poor) to 5 (excellent):
- factual_accuracy: Are claims appropriately hedged? Are UNVERIFIED or CONTRADICTED
  claims clearly flagged rather than stated as settled fact? Penalize overconfidence.
- completeness: Does the report substantively address the original query, covering
  the sub-topics a thorough answer would need?
- readability: Is it well-structured, non-redundant, and clearly written?

Respond ONLY with a JSON object:
{"factual_accuracy": <1-5>, "completeness": <1-5>, "readability": <1-5>, "justification": "<2-3 sentences>"}
"""


@dataclass
class EvalResult:
    query: str
    factual_accuracy: Optional[int]
    completeness: Optional[int]
    readability: Optional[int]
    justification: str
    report_length_chars: int
    duration_seconds: float
    error: Optional[str] = None


def judge_report(query: str, report_markdown: str, judge_llm=None) -> dict:
    judge_llm = judge_llm or get_llm()
    raw = judge_llm.chat(
        system=JUDGE_SYSTEM_PROMPT,
        user=f"Original query: {query}\n\nGenerated report:\n{report_markdown}\n\nReturn the JSON scorecard now.",
        max_tokens=500,
    )
    from agents.base_agent import BaseAgent
    parsed = BaseAgent.parse_json(raw)
    if not isinstance(parsed, dict):
        return {"factual_accuracy": None, "completeness": None, "readability": None,
                 "justification": f"judge output unparseable: {raw[:200]}"}
    return parsed


def run_evaluation(queries: List[str], mode: str = "hierarchical") -> List[EvalResult]:
    orchestrator = SentinelSearchOrchestrator(mode=mode)
    judge_llm = get_llm()
    results = []

    for query in queries:
        print(f"\n=== Running: {query}")
        start = time.time()
        try:
            run_result = orchestrator.run(query, status_cb=lambda m: print(f"  {m}"))
            duration = time.time() - start

            if "error" in run_result:
                results.append(EvalResult(query, None, None, None, "", 0, duration, error=run_result["error"]))
                continue

            report = run_result["report_markdown"]
            print("  Judging report...")
            scores = judge_report(query, report, judge_llm=judge_llm)
            results.append(EvalResult(
                query=query,
                factual_accuracy=scores.get("factual_accuracy"),
                completeness=scores.get("completeness"),
                readability=scores.get("readability"),
                justification=scores.get("justification", ""),
                report_length_chars=len(report),
                duration_seconds=round(duration, 1),
            ))
            print(f"  Scores: {scores}")
        except Exception as e:
            results.append(EvalResult(query, None, None, None, "", time.time() - start, error=str(e)))

    return results


def save_results(results: List[EvalResult], out_dir: str = "./eval/results"):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"eval_{ts}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = os.path.join(out_dir, f"eval_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    valid = [r for r in results if r.factual_accuracy is not None]
    if valid:
        avg_acc = sum(r.factual_accuracy for r in valid) / len(valid)
        avg_comp = sum(r.completeness for r in valid) / len(valid)
        avg_read = sum(r.readability for r in valid) / len(valid)
        print(f"\n=== Summary over {len(valid)}/{len(results)} successful runs ===")
        print(f"Avg factual_accuracy: {avg_acc:.2f}/5")
        print(f"Avg completeness:     {avg_comp:.2f}/5")
        print(f"Avg readability:      {avg_read:.2f}/5")

    print(f"\nSaved: {json_path}\nSaved: {csv_path}")
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation for SentinelSearch")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N sample queries")
    parser.add_argument("--queries", nargs="*", default=None, help="Custom queries instead of the built-in sample set")
    parser.add_argument("--mode", choices=["sequential", "hierarchical"], default="hierarchical")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_SAMPLE_QUERIES
    if args.n:
        queries = queries[:args.n]

    results = run_evaluation(queries, mode=args.mode)
    save_results(results)


if __name__ == "__main__":
    main()
