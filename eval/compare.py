"""
Compares LLM-judge scores against a human's own manual scores for the same
reports — the plan's explicit "compare judge scores to human assessments
for 5+ sample queries" requirement.

Workflow:
    1. Run `python -m eval.judge` — produces eval/results/eval_<ts>.csv
    2. Copy eval/human_scores_template.csv to eval/human_scores.csv and fill
       in your own 1-5 scores per query after reading each generated report
    3. Run `python -m eval.compare eval/results/eval_<ts>.csv eval/human_scores.csv`
"""
from __future__ import annotations
import argparse
import csv
from typing import Dict


def load_scores(path: str) -> Dict[str, dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r["query"]: r for r in rows}


def compare(judge_csv: str, human_csv: str):
    judge_scores = load_scores(judge_csv)
    human_scores = load_scores(human_csv)

    dims = ["factual_accuracy", "completeness", "readability"]
    print(f"{'Query':<55} | {'Dim':<16} | {'Judge':>5} | {'Human':>5} | {'Diff':>5}")
    print("-" * 100)

    total_diff, n = 0, 0
    for query, jrow in judge_scores.items():
        hrow = human_scores.get(query)
        if not hrow:
            print(f"{query[:53]:<55} | (no human score found for this query)")
            continue
        for dim in dims:
            try:
                j = float(jrow[dim])
                h = float(hrow[dim])
            except (ValueError, KeyError):
                continue
            diff = abs(j - h)
            total_diff += diff
            n += 1
            print(f"{query[:53]:<55} | {dim:<16} | {j:>5.1f} | {h:>5.1f} | {diff:>5.1f}")

    if n:
        print("-" * 100)
        print(f"Mean absolute difference (judge vs. human) across {n} scored dimensions: {total_diff / n:.2f}")
        print("Lower is better agreement. A value under ~1.0 on a 5-point scale is generally considered good alignment.")


def main():
    parser = argparse.ArgumentParser(description="Compare LLM-judge scores to human scores")
    parser.add_argument("judge_csv", help="Path to eval/results/eval_<ts>.csv produced by eval.judge")
    parser.add_argument("human_csv", help="Path to your filled-in human_scores.csv")
    args = parser.parse_args()
    compare(args.judge_csv, args.human_csv)


if __name__ == "__main__":
    main()
