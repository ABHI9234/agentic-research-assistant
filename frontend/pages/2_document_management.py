import streamlit as st
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from frontend.utils.api_client import upload_document, get_job_status, list_documents
import httpx

st.set_page_config(page_title="Document Management", page_icon="📁", layout="wide")
st.title("📁 Document Management")

# ── Danger zone — clear all data ─────────────────────────────────────────────
with st.expander("🗑️ Clear All Data (New Session)"):
    st.warning("This permanently deletes all documents, chunks, and knowledge graph data from Qdrant and Neo4j.")
    if st.button("⚠️ Wipe Everything & Start Fresh", type="secondary"):
        with st.spinner("Clearing all data..."):
            try:
                r = httpx.delete("http://localhost:8000/api/ingest/clear-all", timeout=30)
                result = r.json()
                if result.get("status") == "cleared":
                    st.success("✅ All data cleared. You can now upload fresh documents.")
                else:
                    st.warning(f"Partial clear: {result}")
            except Exception as e:
                st.error(f"Clear failed: {e}")

st.divider()

# ── Upload section ────────────────────────────────────────────────────────────
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Drag and drop PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("⬆️ Upload & Ingest", type="primary"):
    for uploaded_file in uploaded_files:
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            result = upload_document(uploaded_file.read(), uploaded_file.name)
        if "error" in result:
            st.error(f"{uploaded_file.name}: {result['error']}")
        else:
            job_id = result.get("job_id")
            st.success(f"{uploaded_file.name} uploaded. Job ID: `{job_id}`")
            progress_bar = st.progress(0)
            status_text = st.empty()
            for _ in range(60):
                status = get_job_status(job_id)
                pct = int(status.get("progress_percent", 0))
                progress_bar.progress(pct)
                status_text.text(f"Status: {status.get('status')} ({pct}%)")
                if status.get("status") in ("completed", "failed"):
                    break
                time.sleep(3)
            if status.get("status") == "completed":
                st.success(
                    f"✅ Done — {status.get('chunks_created', 0)} chunks, "
                    f"{status.get('entities_extracted', 0)} entities"
                )
            else:
                st.error(f"❌ Failed: {status.get('error_message', 'Unknown error')}")

st.divider()

# ── Document list ─────────────────────────────────────────────────────────────
st.subheader("Ingested Documents")
if st.button("🔄 Refresh"):
    st.rerun()

docs_data = list_documents()
docs = docs_data.get("documents", [])

if not docs:
    st.info("No documents ingested yet. Upload a PDF above.")
else:
    import pandas as pd
    df = pd.DataFrame([{
        "Filename": d["filename"],
        "Status": d["status"],
        "Chunks": d["chunk_count"],
        "Entities": d["entity_count"],
        "Size (KB)": d["file_size_kb"],
        "Uploaded": d["upload_date"][:19] if d["upload_date"] else "",
    } for d in docs])
    st.dataframe(df, use_container_width=True)
    st.caption(f"Total: {docs_data.get('total', 0)} documents")
