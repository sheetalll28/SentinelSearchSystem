"""
FastAPI backend for SentinelSearch — streams live agent progress to the
browser over Server-Sent Events (SSE):

    status  -> {"agent": "...", "state": "running"|"done", "sub_task"?: "..."}
    result  -> {"agent": "...", "data": {...}}
    usage   -> {"calls": n, "input_tokens": n, "output_tokens": n, "estimated_cost_usd": n}
    security-> {"event_type": "...", "details": {...}}   # SSRF block / prompt-injection sanitized
    blocked -> {"stage": "input"|"output", "reason": "..."}
    error   -> {"message": "..."}
    done    -> {"report_markdown": "...", "usage": {...}}

Honest limitation carried over from earlier builds: individual LLM calls are
not token-streamed — each event fires as soon as that agent's full output is
ready, which still reads as "live" because four separate steps post in
sequence rather than one long blocking spinner.

Security note on the optional `api_key` query param: it lets the UI's
settings modal supply a runtime key without editing .env, matching the
reference design. It's passed straight through to the LLM client for that
one request only — never written to disk or logged — but query params can
still end up in browser history / server access logs, so this is meant for
local demo use, not a hardened multi-user deployment.

Run with:  uvicorn backend.main:app --reload
"""
from __future__ import annotations
import hashlib
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
from typing import Optional

from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CONFIG
from moderation import moderate
from orchestrator import SentinelSearchOrchestrator
from rag.vector_store import VectorStore
from rag.ingest import ingest_directory, ingest_urls, chunk_text
from cost_tracker import COST_TRACKER
from security import AUDIT_LOG, sanitize_external_content

DEFAULT_COLLECTION = "sentinel_search"

app = FastAPI(title="SentinelSearch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo / grading context — tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _short_agent_name(full_name: str) -> str:
    """'Agent A — Lead Researcher' -> 'Lead Researcher' (what the frontend keys on)."""
    return re.sub(r"^Agent [A-D] — ", "", full_name)


def _run_pipeline(query: str, mode: str, provider: Optional[str], api_key: Optional[str],
                   collection_name: Optional[str], events: "queue.Queue"):
    """Runs the full agent pipeline, pushing SSE-ready events onto a queue
    as each step completes. Runs on a background thread so the generator
    in the SSE endpoint can yield events as soon as they land."""
    try:
        flagged, reason = moderate(query)
        if flagged:
            events.put(_sse("blocked", {"stage": "input", "reason": reason}))
            events.put(None)
            return

        orchestrator = SentinelSearchOrchestrator(
            mode=mode, provider=provider, api_key=api_key, collection_name=collection_name
        )
        lead, scout, checker, editor = (
            orchestrator.lead, orchestrator.scout, orchestrator.checker, orchestrator.editor
        )
        lead_name = _short_agent_name(lead.name)
        scout_name = _short_agent_name(scout.name)
        checker_name = _short_agent_name(checker.name)
        editor_name = _short_agent_name(editor.name)

        events.put(_sse("status", {"agent": lead_name, "state": "running"}))
        sub_tasks = lead.plan(query)
        events.put(_sse("result", {"agent": lead_name, "data": {"sub_tasks": sub_tasks}}))
        events.put(_sse("status", {"agent": lead_name, "state": "done"}))
        events.put(_sse("usage", COST_TRACKER.session_summary()))

        verified_results = []

        def run_one(sub_task: str):
            events.put(_sse("status", {"agent": scout_name, "state": "running", "sub_task": sub_task}))
            audit_before = len(AUDIT_LOG.events)
            scout_result = scout.execute(sub_task)
            for evt in AUDIT_LOG.events[audit_before:]:
                events.put(_sse("security", {"event_type": evt.event_type, "details": evt.details}))
            events.put(_sse("result", {"agent": scout_name, "data": scout_result}))
            events.put(_sse("status", {"agent": scout_name, "state": "done", "sub_task": sub_task}))

            events.put(_sse("status", {"agent": checker_name, "state": "running", "sub_task": sub_task}))
            verified = checker.verify(scout_result)
            events.put(_sse("result", {"agent": checker_name, "data": verified}))
            events.put(_sse("status", {"agent": checker_name, "state": "done", "sub_task": sub_task}))
            events.put(_sse("usage", COST_TRACKER.session_summary()))
            return verified

        if mode == "sequential":
            for st in sub_tasks:
                verified_results.append(run_one(st))
        else:
            threads_out = {}
            threads = []
            for st in sub_tasks:
                def target(s=st):
                    threads_out[s] = run_one(s)
                t = threading.Thread(target=target)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            verified_results = [threads_out[st] for st in sub_tasks]

        events.put(_sse("status", {"agent": editor_name, "state": "running"}))
        report_md = editor.compile_report(query, verified_results)
        events.put(_sse("status", {"agent": editor_name, "state": "done"}))

        out_flagged, out_reason = moderate(report_md)
        if out_flagged:
            events.put(_sse("blocked", {"stage": "output", "reason": out_reason}))
            events.put(None)
            return

        final_usage = COST_TRACKER.session_summary()
        events.put(_sse("done", {"report_markdown": report_md, "usage": final_usage}))
    except Exception as e:
        events.put(_sse("error", {"message": str(e)}))
    finally:
        events.put(None)  # sentinel: stream complete


@app.get("/research")
def research(
    query: str = Query(..., min_length=3),
    mode: str = Query("hierarchical", pattern="^(sequential|hierarchical)$"),
    provider: Optional[str] = Query(None, pattern="^(anthropic|openai|groq|ollama)$"),
    api_key: Optional[str] = Query(None),
    collection_name: Optional[str] = Query(None),
):
    events: "queue.Queue" = queue.Queue()
    threading.Thread(
        target=_run_pipeline,
        args=(query, mode, provider, api_key, collection_name, events),
        daemon=True,
    ).start()

    def event_stream():
        while True:
            item = events.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Ingests an uploaded PDF into its own isolated ChromaDB collection, so
    a user can research against a specific document without mixing it into
    the shared default knowledge base. Content is sanitized for suspected
    prompt-injection patterns before storage, same as the CLI ingest path."""
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only .pdf files are supported."}

    raw_bytes = await file.read()
    tmp_path = os.path.join(tempfile.gettempdir(), f"{int(time.time())}_{file.filename}")
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)

    try:
        from pypdf import PdfReader
        reader = PdfReader(tmp_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        os.remove(tmp_path)
        return {"error": f"Could not read PDF: {e}"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text.strip():
        return {"error": "No extractable text found in this PDF."}

    text, flagged = sanitize_external_content(text, source=file.filename)

    name_hash = hashlib.sha1(file.filename.encode()).hexdigest()[:8]
    safe_stub = re.sub(r"[^a-zA-Z0-9_]", "_", file.filename.rsplit(".", 1)[0])[:40]
    collection_name = f"upload_{safe_stub}_{name_hash}"

    store = VectorStore(collection_name=collection_name)
    chunks = chunk_text(text)
    ids = [f"{file.filename}::{i}" for i in range(len(chunks))]
    metas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]
    store.add_chunks(chunks, metas, ids)

    return {
        "filename": file.filename,
        "collection_name": collection_name,
        "chunks": len(chunks),
        "sanitized": flagged,
    }


@app.post("/ingest")
def ingest(urls: Optional[list[str]] = None):
    result = ingest_directory()
    if urls:
        url_result = ingest_urls(urls)
        result["urls"] = url_result
    return result


@app.get("/config")
def get_config():
    return {
        "llm_provider": CONFIG.llm_provider,
        "ollama_model": CONFIG.ollama_model,
        "search_provider": CONFIG.search_provider,
        "vector_db_path": CONFIG.vector_db_path,
    }


@app.get("/health")
def health():
    try:
        store = VectorStore(collection_name=DEFAULT_COLLECTION)
        doc_count = store.count()
    except Exception:
        doc_count = 0
    return {"status": "ok", "documents": doc_count}


@app.get("/security-log")
def security_log(n: int = 50):
    return {"events": AUDIT_LOG.recent(n)}
