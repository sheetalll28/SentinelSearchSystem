# 🛰️ SentinelSearch

**An autonomous, provider-agnostic multi-agent research system.**
SentinelSearch takes a research query, breaks it down into smaller tasks, searches the live web for relevant information, cross-checks the findings against a local knowledge base using RAG, and finally generates a structured Markdown report with citations.

The system uses four AI agents, with each agent handling a different part of the research process.

This implementation also extends the original internship prototype in two important ways:

1. **LLM-agnostic** — It works with Claude, OpenAI, or a fully local/offline Ollama model. The provider can be changed using a single `.env` variable without changing the code.
2. **Corpus-agnostic RAG** — The system can ingest any collection of PDFs/TXT files, or selected URLs, placed inside `data/documents/`. This avoids depending on one particular document or dataset.

---

## Architecture

```text
User Query
   │
   ▼
Agent A — Lead Researcher   (deconstructs query → 3-5 sub-tasks)
   │
   ├──▶ Agent B — Web Scout      (searches the web, extracts claims)
   │        │
   │        ▼
   └──▶ Agent C — Fact Checker   (cross-verifies claims vs. local RAG store)
            │
            ▼
         Agent D — Editor          (synthesises verified findings → Markdown report)
            │
            ▼
      Structured Markdown Report (with citations)
```
## SentinelSearch — LLM Abstraction Architecture
                         ┌──────────────────────┐
                         │   SentinelSearch      │
                         │       Agents          │
                         │                      │
                         │ Research Agent       │
                         │ Analysis Agent       │
                         │ Evidence Agent       │
                         │ Report Agent         │
                         └──────────┬───────────┘
                                    │
                                    │ llm.chat(
                                    │   system,
                                    │   user,
                                    │   max_tokens
                                    │ )
                                    ▼
                         ┌──────────────────────┐
                         │       BaseLLM        │
                         │   Abstract Interface │
                         │                      │
                         │ + chat(...)          │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │  ClaudeLLM   │  │  OpenAILLM   │  │   GroqLLM    │
          │              │  │              │  │              │
          │ Anthropic SDK│  │ OpenAI SDK   │  │ OpenAI SDK   │
          │              │  │              │  │ + Groq URL   │
          └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                 │                 │                 │
                 ▼                 ▼                 ▼
             Claude API        OpenAI API         Groq API

                                    │
                                    │
                              ┌─────┴─────┐
                              │ OllamaLLM │
                              │           │
                              │ Ollama SDK│
                              │ Local LLM │
                              └───────────┘


                  ┌─────────────────────────────┐
                  │          get_llm()          │
                  │                             │
                  │ Reads provider/config       │
                  │ and creates correct object  │
                  └──────────────┬──────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
             anthropic         openai           groq
                │                │                │
                ▼                ▼                ▼
             ClaudeLLM        OpenAILLM        GroqLLM
                                      
                              ollama
                                 │
                                 ▼
                             OllamaLLM


        ┌────────────────────────────────────────────────┐
        │                 Supporting Layer               │
        │                                                │
        │  CONFIG                 COST_TRACKER           │
        │  ├─ API Keys             ├─ Provider          │
        │  ├─ Model Names          ├─ Model             │
        │  ├─ Provider             ├─ Input Tokens      │
        │  └─ Ollama Host          └─ Output Tokens     │
        └────────────────────────────────────────────────┘

```

There are two orchestration modes available through the `--mode` flag:

* **sequential** — The sub-tasks are handled one after another: Agent A → B → C → D.
* **hierarchical** (default) — The sub-tasks are distributed and Agents B/C can work concurrently before Agent D brings the results together.

## Project Layout

```text
sentinel_search/
├── config.py              # Loads .env into a unified Config object
├── llm_providers.py       # Claude / OpenAI / Ollama behind a single interface
├── search_tools.py        # DuckDuckGo (free) / Tavily / Serper unified interface
├── moderation.py          # Input/output content moderation gate
├── cost_tracker.py        # Token usage and estimated USD cost per query
├── orchestrator.py        # Wires agents A→B→C→D, supporting sequential or hierarchical flow (CLI-ready)
├── cli.py                 # Command-line entrypoint
├── backend/main.py        # FastAPI server — streams live agent progress via SSE
├── frontend/index.html    # Dark-themed live UI (communicates with backend/main.py)
├── app_streamlit.py       # Streamlit alternative UI (simpler, lower maintenance)
├── rag/
│   ├── ingest.py           # Chunking and generalized document/URL ingestion
│   └── vector_store.py     # ChromaDB abstraction layer
├── agents/
│   ├── base_agent.py       # Shared LLM-call and JSON-parsing utilities
│   ├── lead_researcher.py  # Agent A
│   ├── web_scout.py        # Agent B
│   ├── fact_checker.py     # Agent C
│   └── editor.py           # Agent D
├── tests/                 # Unit tests (RAG chunking, agent logic; LLM calls mocked)
├── data/documents/        # ← Add your PDFs/TXT files here
├── .env.example
└── requirements.txt
```

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

* `LLM_PROVIDER=anthropic|openai|ollama` along with the corresponding API key/model.
* `SEARCH_PROVIDER=duckduckgo` (free, no key) `| tavily | serper`
* For Ollama, start `ollama serve` and run `ollama pull llama3.1` locally first.

## Usage

**1. Build the knowledge base** by adding your PDFs/TXT files to `data/documents/`:

```bash
python cli.py ingest
# Or also ingest specific web pages into the same store:
python cli.py ingest --urls https://example.com/article1 https://example.com/article2
```

**2. Run a research query from the CLI:**

```bash
python cli.py run "What are the tradeoffs between RAG and fine-tuning?" \
    --mode hierarchical --out report.md
```

**3. Or launch the interactive web UI (FastAPI backend + custom frontend):**

```bash
uvicorn backend.main:app --reload
```

Then open `frontend/index.html` directly in your browser. It will connect automatically to `http://127.0.0.1:8000`.

The UI shows the progress of the four agents through **Server-Sent Events**. The Lead Researcher, Web Scout, Fact Checker, and Editor cards update as each stage finishes. Once the research is complete, the generated Markdown report is displayed along with copy/download buttons and a running token/cost tracker.

Additional features include:

* **Runtime provider switching** — The settings panel lets you choose Ollama, Claude, or OpenAI and provide an API key for that request without changing `.env`. The API key is sent as a URL query parameter to the local backend for that request only and is not persisted or logged. This is intended for local development and should not be exposed beyond the local machine.
* **Per-document RAG** — Uploading a PDF from the sidebar creates a separate ChromaDB collection for that document. This allows the Fact Checker to use that particular document as its reference instead of the default knowledge base until you switch back.
* **Security indicator** — The UI shows when Web Scout content triggers SSRF or prompt-injection checks in `security.py`.

One limitation is that individual LLM calls are not streamed word-by-word. Each event is shown once an agent has finished generating its complete output. Even so, having four separate stages makes the progress easier to follow compared with a single long loading screen.

**4. Or use the streamlined Streamlit interface:**

```bash
streamlit run app_streamlit.py
```

This uses the same pipeline with a simpler interface and less frontend code to maintain. It can be useful if you prefer running everything through a single process.

## Security

Since the system allows agents to fetch web content and pass that content to an LLM, there are some additional security risks to consider. Two of the main ones are handled in `security.py`:

* **SSRF protection** — Every URL that Web Scout or the RAG ingester tries to access is checked using `is_safe_url()` before the request is made. It blocks non-http(s) schemes and checks whether the hostname resolves to loopback, private/LAN, link-local, or cloud metadata addresses such as `169.254.169.254`. This helps prevent malicious or compromised search results from being used for SSRF attacks.
* **Prompt-injection defense** — Scraped web pages and ingested documents are checked using `detect_prompt_injection()` for common prompt-injection phrases such as "ignore previous instructions" or "reveal your system prompt". When a match is found, `sanitize_external_content()` redacts the suspicious part while keeping the surrounding content instead of removing the entire source.
* **Audit log** — Blocked requests and detected injection attempts are recorded in `data/security_log.jsonl` through `AUDIT_LOG`. These events are also shown in the FastAPI UI through `security` SSE events, making it possible to review what happened after a run.
* These checks work alongside the existing `moderation.py` gate for the user's query and the final generated report.

The security checks are covered in `tests/test_security.py`, including SSRF blocking, injection detection, and clean-text handling.

## LLM-as-a-Judge Evaluation

As part of Phase 5, `eval/judge.py` runs the complete pipeline on five sample queries and uses a separate LLM to score each generated report from 1–5.

The reports are evaluated on:

* **factual_accuracy** — Checks whether unverified or conflicting claims are clearly marked instead of being presented as confirmed facts. The evaluation favors honest uncertainty and penalizes overconfident answers.
* **completeness** — Checks whether the report covers the research query properly.
* **readability** — Looks at the structure, clarity, organization, and unnecessary repetition in the report.

```bash
python -m eval.judge                # Run all 5 built-in sample queries
python -m eval.judge --n 2           # Quick run: first 2 queries only
```

Results are saved to `eval/results/eval_<timestamp>.{json,csv}`, and an overall summary is printed in the console.

**To compare the judge scores with a human evaluation**, copy `eval/human_scores_template.csv` to `eval/human_scores.csv`, read the generated reports, add your own 1–5 scores, and then run:

```bash
python -m eval.compare eval/results/eval_<timestamp>.csv eval/human_scores.csv
```

This gives a per-dimension comparison between the automated judge and human scores, along with the mean absolute difference. This helps check whether the automated evaluation is actually aligned with human judgment instead of simply giving consistently high scores.

## Docker (Bonus)

```bash
cp .env.example .env   # Fill in your API keys first
docker compose up --build
```

The API will then be available at `http://127.0.0.1:8000`. Open `frontend/index.html` in the browser as usual; it connects to the same port. The `data/` directory is mounted as a volume so that the vector store and logs remain available after container restarts.

**Note on testing:** The `Dockerfile` and `docker-compose.yml` follow standard practices, including a slim Python base image, `pip install -r requirements.txt`, running `uvicorn` on port 8000, and mounting the `data/` directory as a volume. However, I have not verified the Docker build in this environment, so it should be tested locally with `docker compose up --build` before using it for a production deployment.

## Testing

```bash
pytest tests/ -v
```

The tests cover the document chunking logic, including size limits, overlap, and edge cases. They also cover agent tool-calling logic such as JSON plan parsing, fallback handling for malformed output, and web-search invocation.

LLM and search calls are mocked, so the tests can be run without API keys or an active network connection.

## Known Limitations & Failure Modes

* **Prompt-injection detection is pattern-based, not semantic** — The current implementation looks for known patterns defined in `security.py`. A sufficiently reworded injection could still bypass the checks. So this should be treated as a useful security layer, not a complete solution.

* **LLM JSON drift** — Sub-task plans and claim/verdict lists depend on the LLM returning valid JSON. `BaseAgent.parse_json` has fallback handling for fenced or prose-wrapped JSON, but a poorly behaving model can still result in a best-effort single-item list.

* **Fact-checking is bounded by the local RAG corpus** — If `data/documents/` is empty or does not contain relevant information, Agent C has less grounding available and may have to rely more on general model knowledge. Such claims are explicitly marked as `UNVERIFIED` in the report.

* **DuckDuckGo (default free search)** — DuckDuckGo does not have an official API and may rate-limit requests or return limited results under heavy usage. For production use, `SEARCH_PROVIDER` can be changed to `tavily` or `serper`.

* **Moderation** — If an `OPENAI_API_KEY` is not available for the official Moderation API, the system uses a lightweight local keyword-based heuristic. This is useful as a fallback but should not be treated as a replacement for a production-grade moderation system.

* **Cost estimates** — The cost tracker uses hardcoded per-million-token pricing in `cost_tracker.py`. These values will need to be updated if the providers change their pricing.

* **Ollama mode** — Running Ollama locally requires enough RAM/VRAM for the selected model. If the machine has limited resources, a smaller model such as `llama3.1:8b` or smaller may be a better choice.

## Deliverables Checklist (Mapped to Plan Grading Rubric)

| Phase       | Requirement                                                                         | Status                                                                                                                                    |
| ----------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1           | Prompt library (3 personas) + comparative analysis                                  | Superseded — Agent personas are implemented in `agents/*.py`; provider comparison is available through the `--provider` flag on any query |
| 2           | Local RAG system, chunking strategy, grounded/cited answers                         |  `rag/ingest.py`, `rag/vector_store.py` — generalized to work with any folder rather than one fixed file                                 |
| 3           | Single-agent CLI tool, ReAct pattern, robustness                                    |  `cli.py run`, with retry logic in `llm_providers.py`/`search_tools.py`                                                                  |
| 4           | Multi-agent MVP, sequential + hierarchical flow, shared state                       |  `orchestrator.py`, with both modes tested                                                                                               |
| 5           | LLM-as-a-judge eval vs. human, UI w/ live status, moderation, cost tracking, README |  `eval/judge.py` + `eval/compare.py`, FastAPI+SSE UI, `moderation.py`, `cost_tracker.py`                                                 |
 
## Tech Stack

| Layer         | Technology                                  |
| ------------- | ------------------------------------------- |
| Language      | Python 3.11+                                |
| LLM Access    | Anthropic API / OpenAI API / Ollama (local) |
| RAG Framework | Custom chunking + ChromaDB                  |
| Vector DB     | ChromaDB (local, persistent)                |
| Search Tool   | DuckDuckGo (free) / Tavily / Serper         |
| Frontend      | Streamlit                                   |
| Testing       | pytest                                      |
