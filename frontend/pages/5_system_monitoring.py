import streamlit as st
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from frontend.utils.api_client import get_detailed_health, get_metrics

st.set_page_config(page_title="System Monitoring", page_icon="🖥️", layout="wide")
st.title("🖥️ System Monitoring")

auto_refresh = st.toggle("Auto-refresh every 10s", value=False)

def render_status(label, status):
    if status == "healthy":
        st.success(f"✅ {label}: healthy")
    elif "unhealthy" in str(status):
        st.error(f"❌ {label}: {status}")
    else:
        st.warning(f"⚠️ {label}: {status}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Service Health")
    with st.spinner("Checking services..."):
        health = get_detailed_health()
    services = health.get("services", {})
    render_status("API", services.get("api", "unknown"))
    render_status("Qdrant", services.get("qdrant", "unknown"))
    render_status("Neo4j", services.get("neo4j", "unknown"))
    render_status("Groq", services.get("groq", "unknown"))
    st.caption(f"Overall: {health.get('status','?')} | Uptime: {health.get('uptime_seconds',0):.0f}s")

with col2:
    st.subheader("System Metrics")
    metrics = get_metrics()
    if "error" not in metrics:
        st.metric("CPU %", f"{metrics.get('cpu_percent', 0):.1f}%")
        st.metric("Memory Used", f"{metrics.get('memory_used_mb', 0):.0f} MB")
        st.metric("Memory Total", f"{metrics.get('memory_total_mb', 0):.0f} MB")
        st.metric("Memory %", f"{metrics.get('memory_percent', 0):.1f}%")
    else:
        st.warning("Metrics unavailable — is the backend running?")

if auto_refresh:
    time.sleep(10)
    st.rerun()
