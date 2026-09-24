"""
Generalized ingestion pipeline. Unlike the internship prototype (which only
read one hardcoded PDF), this walks an entire directory of mixed PDF/TXT
files, and can optionally ingest a list of URLs — so the RAG system works
on whatever corpus a user drops into data/documents/.
"""
from __future__ import annotations
import os
import re
from typing import List, Dict
from pypdf import PdfReader

from config import CONFIG
from rag.vector_store import VectorStore
from search_tools import fetch_page_text
from security import sanitize_external_content


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_txt(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Recursive-ish character chunking with overlap (Phase 2 requirement)."""
    chunk_size = chunk_size or CONFIG.chunk_size
    overlap = overlap or CONFIG.chunk_overlap
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # try to break on a sentence boundary near the end
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + chunk_size * 0.5:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) if overlap >= end - start else end - overlap
        if start <= 0 or end == len(text):
            if end == len(text):
                break
    return [c for c in chunks if c]


def ingest_directory(directory: str = None, store: VectorStore = None) -> Dict:
    """Ingest every .pdf and .txt file found in `directory` (recursively)."""
    directory = directory or CONFIG.documents_path
    store = store or VectorStore()

    if not os.path.isdir(directory):
        return {"files_processed": 0, "chunks_added": 0, "error": f"'{directory}' not found"}

    files_processed, chunks_added = 0, 0
    for root, _, files in os.walk(directory):
        for fname in files:
            ext = fname.lower().rsplit(".", 1)[-1]
            if ext not in ("pdf", "txt"):
                continue
            fpath = os.path.join(root, fname)
            try:
                raw = _read_pdf(fpath) if ext == "pdf" else _read_txt(fpath)
            except Exception as e:
                print(f"  ! skipped {fname}: {e}")
                continue

            raw, flagged = sanitize_external_content(raw, source=fname)
            chunks = chunk_text(raw)
            ids = [f"{fname}::{i}" for i in range(len(chunks))]
            metas = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]
            store.add_chunks(chunks, metas, ids)
            files_processed += 1
            chunks_added += len(chunks)
            flag_note = " [injection pattern sanitized]" if flagged else ""
            print(f"  + {fname}: {len(chunks)} chunks{flag_note}")

    return {"files_processed": files_processed, "chunks_added": chunks_added}


def ingest_urls(urls: List[str], store: VectorStore = None) -> Dict:
    """Ingest arbitrary web pages into the same vector store (extends beyond PDFs)."""
    store = store or VectorStore()
    chunks_added = 0
    for url in urls:
        text = fetch_page_text(url, char_limit=20000)
        if text.startswith("[fetch failed") or text.startswith("[fetch blocked"):
            print(f"  ! {url}: {text}")
            continue
        text, flagged = sanitize_external_content(text, source=url)
        if flagged:
            print(f"  ! {url}: injection pattern sanitized")
        chunks = chunk_text(text)
        ids = [f"{url}::{i}" for i in range(len(chunks))]
        metas = [{"source": url, "chunk_index": i} for i in range(len(chunks))]
        store.add_chunks(chunks, metas, ids)
        chunks_added += len(chunks)
        print(f"  + {url}: {len(chunks)} chunks")
    return {"urls_processed": len(urls), "chunks_added": chunks_added}


if __name__ == "__main__":
    print(f"Ingesting documents from: {CONFIG.documents_path}")
    result = ingest_directory()
    print(result)
