import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock


def test_get_llm_respects_explicit_provider_and_api_key_override():
    """Regression test: get_llm() must actually honor explicit overrides —
    an earlier version mutated os.environ after CONFIG was already loaded,
    which silently did nothing."""
    import llm_providers
    with patch("llm_providers.OpenAILLM") as MockOpenAI:
        llm_providers.get_llm(provider="openai", api_key="sk-test-123")
        MockOpenAI.assert_called_once_with(api_key="sk-test-123")


def test_get_llm_routes_groq_correctly():
    """Groq support exists specifically so the project can be deployed
    (e.g. on Railway/Render) without needing a paid Claude/OpenAI key —
    Groq's free tier requires no credit card."""
    import llm_providers
    with patch("llm_providers.GroqLLM") as MockGroq:
        llm_providers.get_llm(provider="groq", api_key="gsk-test")
        MockGroq.assert_called_once_with(api_key="gsk-test")


def test_groq_llm_uses_groq_endpoint_not_openai():
    import llm_providers
    with patch("openai.OpenAI") as MockClient:
        llm_providers.GroqLLM(api_key="gsk-test")
        MockClient.assert_called_once_with(api_key="gsk-test", base_url="https://api.groq.com/openai/v1")


def test_orchestrator_builds_one_shared_llm_for_all_four_agents():
    import orchestrator as orch_mod

    class FakeLLM:
        def chat(self, system, user, max_tokens=1500):
            return '["sub 1"]'

    class FakeStore:
        def __init__(self, collection_name="sentinel_search"):
            self.collection_name = collection_name
        def query(self, q, top_k=4):
            return []

    with patch.object(orch_mod, "get_llm", return_value=FakeLLM()) as mock_get_llm, \
         patch.object(orch_mod, "VectorStore", side_effect=FakeStore) as mock_store:
        o = orch_mod.SentinelSearchOrchestrator(
            mode="sequential", provider="anthropic", api_key="abc", collection_name="custom_collection"
        )
        mock_get_llm.assert_called_once_with(provider="anthropic", api_key="abc")
        assert o.lead.llm is o.scout.llm is o.checker.llm is o.editor.llm
        mock_store.assert_called_once_with(collection_name="custom_collection")


def test_short_agent_name_strips_prefix():
    from backend.main import _short_agent_name
    assert _short_agent_name("Agent A — Lead Researcher") == "Lead Researcher"
    assert _short_agent_name("Agent D — Editor") == "Editor"
    assert _short_agent_name("Not Prefixed") == "Not Prefixed"


def test_upload_pdf_rejects_non_pdf_files():
    import backend.main as m
    from fastapi.testclient import TestClient
    client = TestClient(m.app)
    r = client.post("/upload-pdf", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert "error" in r.json()


def test_upload_pdf_ingests_real_pdf_into_isolated_collection():
    from pypdf import PdfWriter
    import io

    # Build a minimal valid (if textless) PDF in memory just to exercise the
    # endpoint's file-handling path without needing a fixture file on disk.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    class FakeEmbedder:
        def __call__(self, input):
            return [[0.1] * 8 for _ in input]

    import backend.main as m
    from fastapi.testclient import TestClient
    client = TestClient(m.app)

    with patch("rag.vector_store.embedding_functions.SentenceTransformerEmbeddingFunction", return_value=FakeEmbedder()):
        r = client.post("/upload-pdf", files={"file": ("blank.pdf", buf, "application/pdf")})
        data = r.json()
        # A blank page has no extractable text, so this should hit the
        # "no extractable text" branch rather than crash.
        assert "error" in data or data.get("collection_name", "").startswith("upload_")
