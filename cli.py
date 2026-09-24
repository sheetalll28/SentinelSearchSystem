"""
SentinelSearch CLI

Usage:
    python cli.py ingest                       # ingest data/documents into the RAG store
    python cli.py ingest --urls a.com b.com     # also ingest specific URLs
    python cli.py run "your research query"     # run the full multi-agent pipeline
    python cli.py run "query" --mode sequential --provider openai
"""
from __future__ import annotations
import argparse
import json
import os
from rich.console import Console
from rich.markdown import Markdown

from config import CONFIG
from rag.ingest import ingest_directory, ingest_urls
from orchestrator import SentinelSearchOrchestrator

console = Console()


def cmd_ingest(args):
    console.print(f"[bold cyan]Ingesting documents from {CONFIG.documents_path}...[/]")
    result = ingest_directory()
    console.print(result)
    if args.urls:
        console.print(f"[bold cyan]Ingesting {len(args.urls)} URL(s)...[/]")
        console.print(ingest_urls(args.urls))


def cmd_run(args):
    orchestrator = SentinelSearchOrchestrator(mode=args.mode, provider=args.provider)
    result = orchestrator.run(args.query, status_cb=lambda m: console.print(f"[dim]{m}[/]"))

    if "error" in result:
        console.print(f"[bold red]{result['error']}[/]")
        return

    console.rule("[bold green]Final Report")
    console.print(Markdown(result["report_markdown"]))
    console.rule("Usage")
    console.print(result["usage"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result["report_markdown"])
        console.print(f"[bold green]Saved report to {args.out}[/]")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        console.print(f"[bold green]Saved full trace to {args.json_out}[/]")


def main():
    parser = argparse.ArgumentParser(description="SentinelSearch — autonomous multi-agent research system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents/urls into the RAG vector store")
    p_ingest.add_argument("--urls", nargs="*", default=[], help="Optional list of URLs to ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    p_run = sub.add_parser("run", help="Run the full multi-agent research pipeline")
    p_run.add_argument("query", type=str, help="Research query")
    p_run.add_argument("--mode", choices=["sequential", "hierarchical"], default="hierarchical")
    p_run.add_argument("--provider", choices=["anthropic", "openai", "groq", "ollama"], default=None)
    p_run.add_argument("--out", type=str, default=None, help="Path to save report.md")
    p_run.add_argument("--json-out", dest="json_out", type=str, default=None, help="Path to save full JSON trace")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
