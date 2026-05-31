import streamlit as st
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from frontend.utils.api_client import upload_document, get_job_status, list_documents, clear_all_data
import httpx

st.set_page_config(page_title="Document Management", page_icon="📁", layout="wide")
st.title("📁 Document Management")

# ── Session state ─────────────────────────────────────────────────────────────
if "uploaded_jobs" not in st.session_state:
    st.session_state.uploaded_jobs = []

# ── Clear all data ────────────────────────────────────────────────────────────
with st.expander("🗑️ Clear All Data (New Session)"):
    st.warning("Permanently deletes all documents, chunks, and graph data.")
    if st.button("⚠️ Wipe Everything & Start Fresh", type="secondary"):
        with st.spinner("Clearing all data..."):
            result = clear_all_data()
            if result.get("status") == "cleared":
                st.session_state.uploaded_jobs = []
                st.success("✅ All data cleared.")
            else:
                st.warning(f"Result: {result}")

st.divider()

# ── Upload — auto ingests as soon as file is added ────────────────────────────
st.subheader("Upload Documents")
st.caption("Add a PDF below — it uploads and ingests automatically. Keep files under 5MB for best results.")

uploaded_files = st.file_uploader(
    "Drag and drop PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
    key="file_uploader",
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Check if already uploaded this session
        already_uploaded = any(
            j["filename"] == uploaded_file.name
            for j in st.session_state.uploaded_jobs
        )
        if not already_uploaded:
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                result = upload_document(uploaded_file.read(), uploaded_file.name)

            if "error" in result:
                st.error(f"❌ {uploaded_file.name}: {result['error']}")
            else:
                job_id = result.get("job_id", "")
                st.session_state.uploaded_jobs.append({
                    "job_id": job_id,
                    "filename": uploaded_file.name,
                    "status": "processing",
                    "chunks": 0,
                    "entities": 0,
                })
                st.success(f"✅ {uploaded_file.name} uploaded. Ingestion started in background.")

st.divider()

# ── Live status table ─────────────────────────────────────────────────────────
st.subheader("Ingested Documents")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Refresh Status"):
        # Update status for all jobs
        for job in st.session_state.uploaded_jobs:
            if job["status"] not in ("completed", "failed"):
                status = get_job_status(job["job_id"])
                if "error" not in status:
                    job["status"] = status.get("status", "processing")
                    job["chunks"] = status.get("chunks_created", 0)
                    job["entities"] = status.get("entities_extracted", 0)
        st.rerun()

# Show session jobs
if st.session_state.uploaded_jobs:
    import pandas as pd
    df = pd.DataFrame([{
        "Filename": j["filename"],
        "Status": j["status"],
        "Chunks": j["chunks"],
        "Entities": j["entities"],
        "Job ID": j["job_id"][:8] + "...",
    } for j in st.session_state.uploaded_jobs])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No documents uploaded this session. Add a PDF above.")

st.caption("⚠️ Note: Document list resets when the backend restarts (free tier limitation). Your data in Qdrant and Neo4j persists permanently.")
