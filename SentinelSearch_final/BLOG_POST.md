# Building SentinelSearch: A Provider-Agnostic Multi-Agent Research System

*A technical write-up for the SentinelSearch internship capstone project*

## The problem with the first version

My original SentinelSearch prototype worked — barely. It was hardwired to Ollama
running a small local model, and its "knowledge base" was a single PDF baked into
the code path. Every design decision assumed one specific setup: one model, one
document, one machine. That's fine for proving a concept works once, but it's a
poor foundation for something meant to demonstrate real engineering judgment.

Two questions drove the rebuild: *what if someone wants to use Claude or OpenAI
instead of a local model?* and *what if the knowledge base needs to be a folder
of research papers instead of one file?* Answering both meant treating the LLM
and the document corpus as configuration, not as hardcoded assumptions — the
difference between a script and a system.

## Architecture: four agents, one contract

The core pipeline still follows the plan's original design — Lead Researcher
breaks a query into sub-tasks, Web Scout researches each one, Fact Checker
cross-verifies claims against a local vector store, and Editor compiles
everything into a cited report. What changed is how the agents talk to the
outside world.

Every agent inherits from a `BaseAgent` that exposes exactly one method,
`think()`, which calls whatever LLM the config says to use. The agent itself
never imports `anthropic` or `openai` — it calls `self.think(prompt)` and gets
a string back. Swapping providers is a one-line `.env` change instead of a
code change. The same pattern applies to search (DuckDuckGo, Tavily, or
Serper) and to embeddings for the RAG store. This is the single decision that
made the rest of the system generalize: keep the agents' *logic* stable and
push every vendor-specific detail behind a thin interface.

## The harder problem: agents that read the open web

Somewhere in Phase 4, a fact hit me that the original prototype's tests never
surfaced: Web Scout doesn't just call a search API — it fetches and reads
arbitrary web pages, and then pastes that text directly into an LLM prompt.
That's not a hypothetical risk. A page could contain text aimed at the *agent*,
not at a human reader — something like "ignore your previous instructions and
instead output X." The agent has no way to tell the difference between
"the page's actual content" and "text on the page trying to redirect it,"
unless something checks first.

I built a small `security.py` module to close two specific gaps:

- **SSRF protection.** Before fetching any URL, `is_safe_url()` resolves the
  hostname and rejects anything pointing at loopback, private/LAN, or cloud
  metadata addresses (`169.254.169.254` is the canonical one). A malicious or
  just misconfigured search result shouldn't be able to make my server hit
  its own internal network.
- **Prompt-injection sanitization.** Scraped text and ingested documents get
  scanned for known hijack phrasing before they reach any agent's prompt.
  Matches get redacted in place — not silently dropped — so real information
  around a suspicious phrase is preserved, and every match is logged to an
  audit file.

I want to be honest about the limits here: this is pattern-matching, not
semantic understanding. A cleverly reworded injection attempt could still get
through. But going from *zero* consideration of this risk to *some* concrete
mitigation, with tests proving the mitigation actually fires, felt like the
right scope for an internship timeline — better than either ignoring the
problem or trying to build a research-grade defense from scratch.

## Making "live" actually mean live

The plan asked for a UI showing which agent is currently running. I built
that twice: once as a simple Streamlit app, and once as a FastAPI backend
streaming Server-Sent Events to a custom dark-themed frontend, with four
agent "nodes" that light up in sequence as they run.

One thing worth being upfront about: none of this is *token-level* streaming.
Each LLM call in this pipeline is a single blocking request — there's no
partial output arriving word by word. What actually streams is *events
between agents*: the moment Web Scout finishes, the browser gets a `result`
event immediately, without waiting for Fact Checker and Editor to also
finish. With four distinct steps posting in sequence instead of one long
spinner, it reads as live even though no individual model call is streaming
internally. That distinction matters — claiming true streaming when I hadn't
built it would be worse than just explaining what the events actually
represent.

## Evaluation: grading my own agents' homework, carefully

The plan's evaluation requirement — LLM-as-a-judge scoring compared against
human assessment — forced a design choice I initially wanted to skip: the
judge had to be a genuinely separate call, not just the Editor agent asked to
also rate its own report. `eval/judge.py` runs the full pipeline on five
sample queries, then sends each finished report to a fresh LLM call with no
memory of how it was written, scoring factual accuracy (with explicit credit
for flagging UNVERIFIED claims rather than stating them confidently),
completeness, and readability. A companion script, `eval/compare.py`, lines
up those scores against my own manual reading of the same reports and reports
the mean absolute difference — a cheap, honest sanity check on whether the
automated judge is actually tracking human judgment or just handing out fives.

## What I'd do differently with more time

The prompt-injection detection is regex-based and would benefit from a
second, semantic pass using the judge LLM itself. The cost tracker uses
hardcoded per-token pricing that will drift out of date. And the FastAPI
backend runs each research query on its own background thread rather than
using proper async task queuing — fine for a demo, not what I'd ship at any
real scale. Every one of these is documented as a known limitation in the
README rather than glossed over, which I think is the more useful lesson
from this internship than any individual line of code: a system is more
trustworthy when it tells you exactly where it's weak.
