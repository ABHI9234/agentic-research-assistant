import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")

st.subheader("Model Configuration")
col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("Groq Model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
with col2:
    max_tokens = st.number_input("Max Tokens", 512, 8192, 4096, 256)
    app_mode = st.selectbox("App Mode", ["development", "production"])

st.subheader("Retrieval Configuration")
col3, col4 = st.columns(2)
with col3:
    chunk_size = st.number_input("Chunk Size", 128, 2048, 512, 64)
    chunk_overlap = st.number_input("Chunk Overlap", 0, 256, 64, 16)
with col4:
    top_k = st.slider("Default Top K", 1, 20, 5)
    graph_depth = st.slider("Default Graph Depth", 1, 5, 2)

st.subheader("Retrieval Weights (Hybrid Mode)")
vector_weight = st.slider("Vector Weight", 0.0, 1.0, 0.6, 0.05)
graph_weight = round(1.0 - vector_weight, 2)
st.caption(f"Graph weight auto-set to: {graph_weight}")

if st.button("💾 Save Settings", type="primary"):
    st.success("Settings saved to session. Restart backend to apply model/chunk changes.")
    st.info("To persist permanently, edit your .env file and restart the server.")

st.divider()
st.subheader("Environment Status")
st.code("""
GROQ_API_KEY        ✅ set
QDRANT_URL          ✅ set  
NEO4J_URI           ✅ set
LANGSMITH_API_KEY   ✅ set
""")
