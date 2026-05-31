import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from frontend.utils.api_client import get_graph, traverse_graph

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")
st.title("🕸️ Knowledge Graph Explorer")

tab1, tab2 = st.tabs(["Graph Overview", "Entity Traversal"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    limit = st.slider("Max nodes to display", 20, 200, 80)
    if st.button("Load Graph", type="primary"):
        with st.spinner("Fetching graph data..."):
            data = get_graph(limit=limit)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        col1, col2 = st.columns(2)
        col1.metric("Total Nodes", data.get("total_nodes", 0))
        col2.metric("Total Edges", data.get("total_edges", 0))

        if nodes:
            try:
                from pyvis.network import Network
                import tempfile, pathlib

                net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
                net.barnes_hut()

                type_colors = {
                    "PERSON": "#e74c3c", "ORG": "#3498db",
                    "CONCEPT": "#2ecc71", "LOCATION": "#f39c12",
                    "TECHNOLOGY": "#9b59b6", "EVENT": "#1abc9c",
                }

                for node in nodes[:limit]:
                    color = type_colors.get(node.get("type", "CONCEPT"), "#95a5a6")
                    net.add_node(
                        node["name"],
                        label=node["name"],
                        color=color,
                        title=f"{node.get('type','?')}: {node.get('description','')}",
                        size=10 + min(node.get("mention_count", 1) * 2, 30),
                    )

                for edge in edges[:limit * 3]:
                    try:
                        net.add_edge(
                            edge["source_entity"],
                            edge["target_entity"],
                            title=edge["relationship_type"],
                        )
                    except Exception:
                        pass

                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                    net.save_graph(f.name)
                    html = pathlib.Path(f.name).read_text()

                st.components.v1.html(html, height=620, scrolling=True)

            except Exception as e:
                st.warning(f"Graph rendering failed: {e}")
                st.json({"nodes": len(nodes), "edges": len(edges)})

            # Node table
            import pandas as pd
            df = pd.DataFrame([{
                "Name": n["name"], "Type": n["type"],
                "Mentions": n["mention_count"],
                "Description": n.get("description", "")[:80],
            } for n in nodes])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No entities found. Ingest documents first.")

# ── Tab 2: Traversal ──────────────────────────────────────────────────────────
with tab2:
    entity_name = st.text_input("Start Entity Name", placeholder="e.g. transformer")
    depth = st.slider("Traversal Depth", 1, 4, 2)

    if st.button("Traverse", type="primary") and entity_name:
        with st.spinner(f"Traversing from '{entity_name}'..."):
            result = traverse_graph(entity_name, depth)

        nodes = result.get("nodes", [])
        paths = result.get("paths", [])

        st.metric("Nodes Reached", len(nodes))

        if paths:
            st.subheader("Paths Found")
            for path in paths[:10]:
                st.markdown(" → ".join(path))

        if nodes:
            import pandas as pd
            df = pd.DataFrame([{"Name": n["name"], "Type": n["type"]} for n in nodes])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No connected entities found. Try a different entity name.")
