"""
Streamlit UI for SentinelSearch.
Run with: streamlit run app_streamlit.py
"""
import streamlit as st
from config import CONFIG
from orchestrator import SentinelSearchOrchestrator
from rag.ingest import ingest_directory

st.set_page_config(page_title="SentinelSearch", page_icon="🛰️", layout="wide")
st.title("🛰️ SentinelSearch — Autonomous Multi-Agent Research")
st.caption(
    f"LLM provider: **{CONFIG.llm_provider}** · Search: **{CONFIG.search_provider}** "
    f"· Vector DB: `{CONFIG.vector_db_path}`"
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Orchestration mode", ["hierarchical", "sequential"], index=0)
    st.divider()
    st.subheader("Knowledge base")
    if st.button("📥 Ingest data/documents/"):
        with st.spinner("Ingesting documents..."):
            result = ingest_directory()
        st.success(f"Processed {result.get('files_processed', 0)} files, "
                    f"{result.get('chunks_added', 0)} chunks added.")
    st.caption("Drop any PDFs/TXT files into `data/documents/` and click ingest — "
               "the RAG store generalizes to whatever corpus you provide.")

query = st.text_input("Research query", placeholder="e.g. What are the tradeoffs of RAG vs fine-tuning?")
run_clicked = st.button("🚀 Run SentinelSearch", type="primary", disabled=not query)

status_placeholder = st.empty()
report_placeholder = st.empty()

if run_clicked and query:
    status_log = []

    def status_cb(msg: str):
        status_log.append(msg)
        status_placeholder.code("\n".join(status_log), language=None)

    orchestrator = SentinelSearchOrchestrator(mode=mode)
    with st.spinner("Agents are working..."):
        result = orchestrator.run(query, status_cb=status_cb)

    if "error" in result:
        st.error(result["error"])
    else:
        report_placeholder.markdown(result["report_markdown"])

        with st.expander("📊 Usage & estimated cost"):
            st.json(result["usage"])

        with st.expander("🔍 Raw agent trace (sub-tasks, claims, verdicts)"):
            st.json(result["results"])

        st.download_button(
            "Download report.md",
            data=result["report_markdown"],
            file_name="sentinel_search_report.md",
            mime="text/markdown",
        )
