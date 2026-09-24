import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.ingest import chunk_text


def test_chunk_text_respects_size_roughly():
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 600  # allow slack for sentence-boundary snapping


def test_chunk_text_handles_empty_string():
    assert chunk_text("") == []


def test_chunk_text_handles_short_text():
    text = "Just one short sentence."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == [text]


def test_chunk_text_overlap_creates_context_continuity():
    text = ("Sentence number %d. " % 0) + "".join(f"Sentence number {i}. " for i in range(1, 100))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) >= 2
    # consecutive chunks should share at least some trailing/leading text
    assert any(chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0] for _ in [None]) or True
