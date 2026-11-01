# 🛰️ SentinelSearch

**A multi-agent research assistant I built on my own to learn how LLM agent systems work end to end.**

You give it a research question. It breaks the question into sub-tasks, searches the web, checks the claims it found against a local knowledge base (RAG), and writes a Markdown report that labels every claim as **SUPPORTED**, **CONTRADICTED** or **UNVERIFIED** instead of presenting everything as fact.

## What I wanted to learn

- How to split a task across several LLM agents with clear roles and strict output formats.
- How RAG really works, so I wrote the chunking, embedding and retrieval steps by hand instead of using a framework.
- How to keep an application independent of any single LLM vendor.
- How to protect an agent that reads untrusted web content (SSRF and prompt injection).
- How to stream live progress to a browser with Server-Sent Events (SSE).
- How to evaluate LLM output, using an LLM judge and checking it against human scores.

## How it works

```text
Research query
   │
   ▼
Input moderation
   │
   ▼
Agent A: Lead Researcher        splits the query into 3 to 5 sub-tasks
   │
   ├── for each sub-task (in parallel, or one by one) ──────────────┐
   │      Agent B: Web Scout       searches the web, extracts claims │
   │             │                                                   │
   │             ▼                                                   │
   │      Agent C: Fact Checker    compares claims with the RAG store│
   │                                                                 │
   ◄─────────────────────────────────────────────────────────────────┘
   │
   ▼
Agent D: Editor                  merges verified findings into one report
   │
   ▼
Output moderation
   │
   ▼
Markdown report (with sources and a token/cost summary)
```

| Agent | Role | Output |
| --- | --- | --- |
| A: Lead Researcher | Deconstructs the query into non-overlapping, web-searchable sub-tasks. | JSON list of up to 5 sub-tasks |
| B: Web Scout | Runs a web search, fetches the top page, removes suspicious text, extracts factual claims. | Sources and a JSON list of claims |
| C: Fact Checker | Retrieves the 4 most relevant chunks from the vector store and judges each claim. | Verdict per claim: SUPPORTED, CONTRADICTED or UNVERIFIED, with a note |
| D: Editor | Writes one structured report, one section per sub-task, and flags claims that are not supported. | Markdown report |

**Orchestration modes** (the `--mode` flag, or the toggle in the UI):

- `hierarchical` (default): the sub-tasks run in parallel threads, then the Editor merges the results.
- `sequential`: the sub-tasks run one after another.

A run makes at most `2N + 2` model calls for `N` sub-tasks: one for the Lead, two per sub-task, and one for the Editor.

## Features

- **Provider-agnostic LLM layer.** Anthropic (Claude), OpenAI, Groq and Ollama (local) sit behind one interface. Switching is one environment variable, a CLI flag or a setting in the UI.
- **RAG over any folder.** Ingest PDFs and TXT files from `data/documents/`, or web pages by URL. In the web UI you can also upload a PDF, which gets its own isolated collection that the Fact Checker uses for that run.
- **Live progress.** The FastAPI backend streams agent status, results, usage and security events over SSE.
- **Safety checks.** SSRF protection, prompt-injection redaction, input and output moderation, and an append-only audit log.
- **Cost tracking.** Token usage and an estimated cost for every call.
- **Evaluation.** An LLM-as-a-judge harness with a script that compares the judge to human scores.
- **Three ways to run it.** Command line, web UI (FastAPI plus a static HTML page), or Streamlit.

## LLM provider layer

Every agent talks to one method, `llm.chat(system, user, max_tokens)`, and never touches a vendor SDK directly.

```text
Agents ──▶ BaseLLM.chat(system, user, max_tokens)
              │
   ┌──────────┼───────────┬───────────┐
   ▼          ▼           ▼           ▼
ClaudeLLM  OpenAILLM    GroqLLM    OllamaLLM
```

- `get_llm(provider, api_key)` is the factory that returns the right client.
- `GroqLLM` reuses the OpenAI SDK and points it at Groq's OpenAI-compatible endpoint. Groq has a free tier, which makes it handy for a public demo.
- Each call is retried up to 3 times with exponential backoff, and every call is logged to the cost tracker.
- An explicit `provider` or `api_key` always overrides the values loaded from `.env`, and the orchestrator builds a single client that all four agents share.

## RAG pipeline

1. **Ingest.** `rag/ingest.py` reads every PDF and TXT file in a folder (or a list of URLs) and removes suspected prompt-injection text.
2. **Chunk.** Fixed windows of 800 characters with 120 characters of overlap, ending on a sentence boundary when one falls past the halfway point.
3. **Embed and store.** `all-MiniLM-L6-v2` (Sentence-Transformers) embeddings are stored in a persistent ChromaDB collection. Chunk IDs are `filename::index` and writes are upserts, so re-ingesting a file overwrites its chunks.
4. **Retrieve.** The Fact Checker queries with the sub-task text and takes the top 4 chunks.

ChromaDB is only touched in `rag/vector_store.py`, so swapping the vector database means rewriting that one file.

## Security

The system fetches web pages and pastes their text into LLM prompts, so `security.py` handles two risks:

- **SSRF.** `is_safe_url()` runs before every fetch. It blocks non-http(s) schemes and any host that resolves to a loopback, private, link-local or reserved address, which includes the cloud metadata address `169.254.169.254`.
- **Prompt injection.** `sanitize_external_content()` scans scraped pages, search snippets and ingested documents for known injection phrases (such as "ignore previous instructions"). Matches are replaced with a marker while the rest of the text is kept.
- **Audit log.** Blocked fetches and detected injections are appended to `data/security_log.jsonl` and also sent to the UI as `security` events.
- **Least privilege.** Agents can search and read, but they cannot write files or run commands.
- **Moderation.** `moderation.py` checks the query on the way in and the report on the way out. It uses OpenAI's Moderation API when `OPENAI_API_KEY` is set, and a small local block-list otherwise.

## Evaluation

`eval/judge.py` runs the full pipeline on five sample queries, then a separate LLM call scores each report from 1 to 5 on:

- **factual_accuracy**: are unverified or contradicted claims flagged instead of stated as fact? Overconfidence is penalized.
- **completeness**: does the report cover the question?
- **readability**: is it clear, organized and free of repetition?

```bash
python -m eval.judge                 # all 5 sample queries
python -m eval.judge --n 2           # first 2 only
```

Results are written to `eval/results/eval_<timestamp>.{json,csv}`. To check the judge against a human, copy `eval/human_scores_template.csv` to `eval/human_scores.csv`, fill in your own 1 to 5 scores, and run:

```bash
python -m eval.compare eval/results/eval_<timestamp>.csv eval/human_scores.csv
```

This prints the per-dimension difference and the mean absolute difference between judge and human.

## Setup

Requires Python 3.11 or newer.

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root. A minimal free setup using Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
SEARCH_PROVIDER=duckduckgo
```

All settings (every one has a default except the API keys):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `groq` or `ollama` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | none, `claude-sonnet-4-6` | Claude settings |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | none, `gpt-4o` | OpenAI settings (the key also enables the OpenAI Moderation API) |
| `GROQ_API_KEY`, `GROQ_MODEL` | none, `llama-3.3-70b-versatile` | Groq settings |
| `OLLAMA_HOST`, `OLLAMA_MODEL` | `http://localhost:11434`, `llama3.1` | Local model (run `ollama serve` and `ollama pull llama3.1` first) |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (free, no key), `tavily` or `serper` |
| `TAVILY_API_KEY`, `SERPER_API_KEY` | none | Keys for those search providers |
| `VECTOR_DB_PATH`, `DOCUMENTS_PATH` | `./data/vector_db`, `./data/documents` | Where the vector store and source documents live |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `800`, `120` | Chunking in characters |
| `ENABLE_MODERATION` | `true` | Turn the moderation gate on or off |

## Usage

**1. Build the knowledge base.** Put PDFs or TXT files in `data/documents/`, then:

```bash
python cli.py ingest
python cli.py ingest --urls https://example.com/article1 https://example.com/article2
```

**2. Run a query from the command line:**

```bash
python cli.py run "What are the tradeoffs between RAG and fine-tuning?" \
    --mode hierarchical --provider groq --out report.md --json-out trace.json
```

**3. Run the web UI:**

```bash
uvicorn backend.main:app --reload
```

Then open `frontend/index.html` in a browser. It connects to `http://127.0.0.1:8000` by default, and the settings panel lets you point it at another backend URL (saved in your browser). The UI shows one card per agent that updates as each stage finishes, a security indicator, a running token and cost counter, and copy and download buttons for the report. You can also choose a provider and enter an API key for a single run, or upload a PDF to fact-check against.

**4. Or run the Streamlit version:**

```bash
streamlit run app_streamlit.py
```

**5. Or use Docker:**

```bash
docker compose up --build
```

The API is then at `http://127.0.0.1:8000`, and `data/` is mounted as a volume so the vector store and logs survive restarts. The container listens on `$PORT` (default 8000). I have not verified the Docker build on every platform, so test it locally before relying on it.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /research` | Runs the pipeline and streams SSE events. Query parameters: `query` (required), `mode`, `provider`, `api_key`, `collection_name`. |
| `POST /upload-pdf` | Ingests a PDF into its own collection and returns the `collection_name` to pass to `/research`. |
| `POST /ingest` | Ingests everything in `data/documents/`, plus optional URLs. |
| `GET /config` | Shows the active provider and settings (no secrets). |
| `GET /health` | Health check with the document count. |
| `GET /security-log` | Recent audit-log events. |

`/research` streams these events: `status`, `result`, `usage`, `security`, `blocked`, `error`, `done` (the final report and usage summary).

Individual LLM calls are not streamed token by token. An event is sent when an agent finishes its whole output, and four separate stages are enough to make progress easy to follow.

## Testing

```bash
pytest tests/ -v
```

LLM and search calls are mocked, so the tests need no API keys or network. They cover chunking, JSON parsing and fallbacks, the Lead Researcher and Web Scout, provider routing, the upload endpoint, the judge's output handling, and the SSRF and injection checks.

## Project layout

```text
├── config.py              # Settings loaded from environment variables
├── llm_providers.py       # Claude / OpenAI / Groq / Ollama behind one interface
├── search_tools.py        # DuckDuckGo / Tavily / Serper search and guarded page fetch
├── security.py            # SSRF check, injection redaction, audit log
├── moderation.py          # Input and output moderation gate
├── cost_tracker.py        # Token usage and estimated cost
├── orchestrator.py        # Wires agents A to D, sequential or hierarchical
├── cli.py                 # Command-line entry point
├── backend/main.py        # FastAPI server with SSE
├── frontend/index.html    # Static UI that listens to the SSE stream
├── app_streamlit.py       # Streamlit alternative UI
├── agents/                # base_agent, lead_researcher, web_scout, fact_checker, editor
├── rag/                   # ingest.py (chunking and ingestion), vector_store.py (ChromaDB)
├── eval/                  # judge.py, compare.py, human_scores_template.csv
├── tests/                 # pytest suite
└── data/                  # documents, vector store and logs
```

## Design choices

- **Fixed pipeline instead of an open agent loop.** The cost and latency of a run are bounded and every stage can be tested on its own. The trade-off is that the system cannot decide to search again when results are thin.
- **UNVERIFIED as the safe default.** If a model's output cannot be parsed, every claim becomes UNVERIFIED, never SUPPORTED.
- **Redact instead of drop.** Suspicious text in a source is replaced in place so the rest of the page is still usable, and the event is logged.
- **One small provider interface.** A single `chat` method works on all four providers, at the cost of not using native tool-calling or streaming.

## Known limitations

- **Verification is only as good as the knowledge base.** The Fact Checker retrieves with the sub-task text rather than each claim and uses the top 4 chunks without a similarity threshold. With an empty or unrelated corpus, most claims come back UNVERIFIED.
- **Citations come from the model.** The Editor is prompted to cite sources as `[n]`, but the numbering is not generated or validated in code.
- **Injection detection is pattern-based.** A reworded injection can slip past it, so treat it as one layer, not a full defense.
- **The SSRF check has gaps.** It does not re-check redirects, and it resolves DNS separately from the request.
- **LLM JSON can drift.** Parsing has fallbacks, but a weak model can still return unusable output, which degrades to best-effort results.
- **Cost tracking is per process.** The tracker is a single global object, so concurrent requests share totals. Prices are hard-coded.
- **Web search.** DuckDuckGo has no official API and may rate-limit. Use `tavily` or `serper` for heavier use.
- **Moderation fallback is small.** The local block-list is a fallback, not a replacement for a real moderation service.
- **Demo-grade API.** There is no authentication or rate limiting, CORS is open, and the optional `api_key` query parameter can end up in logs, so keep the backend local or behind your own protection.

## Ideas for next steps

- Generate and validate citation numbers in code.
- Retrieve per claim, add a relevance threshold, and require agreement from independent sources.
- Re-validate redirects in the SSRF check.
- Keep one failed sub-task from failing the whole run.
- Fall back to another provider when one is down.
- Track cost per request and cancel runs when the browser disconnects.

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| LLMs | Anthropic, OpenAI, Groq, Ollama |
| RAG | Custom chunking, Sentence-Transformers, ChromaDB |
| Search | DuckDuckGo, Tavily, Serper |
| Backend | FastAPI, Uvicorn, SSE |
| Frontend | Static HTML and JavaScript (EventSource), Streamlit alternative |
| Reliability | tenacity retries, audit log, moderation |
| Testing | pytest |
| Packaging | Docker, Docker Compose |
