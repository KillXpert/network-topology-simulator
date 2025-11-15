# app.py — Streamlit Network Topology Simulator (Final Version, Wider Layout)

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrow, Rectangle
import numpy as np
import math
from io import BytesIO
from docx import Document
from docx.shared import Inches
from PIL import Image
import os

# ============================================================================================
# Page Setup
# ============================================================================================
st.set_page_config(page_title="Network Topology Simulator", layout="wide")

# ============================================================================================
# Load images from assets/
# ============================================================================================
ASSET_DIR = "assets"

def load_asset_image(filename):
    """Load image from assets folder or return None."""
    path = os.path.join(ASSET_DIR, filename)
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None

computer_img = load_asset_image("computer.png")
prof_img = load_asset_image("professor.png")
abijith_img = load_asset_image("abijith.jpeg")
dharmayu_img = load_asset_image("dharmyu.jpeg")

# ============================================================================================
# Helper Functions
# ============================================================================================
def hierarchy_pos(G, root, width=2., vert_gap=0.4, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))

    if parent is not None and parent in children:
        children.remove(parent)

    if children:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2

        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos


def draw_double_edge_with_arrows(ax, x1, y1, x2, y2, color="#444"):
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return
    nxp = -dy / L
    nyp = dx / L
    offset = 0.06

    ax.plot([x1 + nxp*offset, x2 + nxp*offset],
            [y1 + nyp*offset, y2 + nyp*offset],
            color=color, linewidth=2)

    ax.plot([x1 - nxp*offset, x2 - nxp*offset],
            [y1 - nyp*offset, y2 - nyp*offset],
            color=color, linewidth=2)

    mid1x, mid1y = (x1 + x2)/2 + nxp*offset, (y1 + y2)/2 + nyp*offset
    mid2x, mid2y = (x1 + x2)/2 - nxp*offset, (y1 + y2)/2 - nyp*offset

    ax.add_patch(FancyArrow(mid1x, mid1y, dx*0.0001, dy*0.0001,
                            width=0.008, head_width=0.05, head_length=0.05, color='black'))
    ax.add_patch(FancyArrow(mid2x, mid2y, -dx*0.0001, -dy*0.0001,
                            width=0.008, head_width=0.05, head_length=0.05, color='black'))


def build_step_by_step(n, num_edges, port_cost, cable_len, cable_cost):
    lines = []
    lines.append("Step-by-step cost calculation:")
    lines.append(f"1) Number of nodes = {n}")
    lines.append(f"2) Number of connections (edges) = {num_edges}")

    total_port_cost = num_edges * 2 * port_cost
    lines.append(f"3) Port cost = {num_edges} * 2 * {port_cost} = {total_port_cost:.2f}")

    total_cable_cost = num_edges * cable_len * cable_cost
    lines.append(f"4) Cable cost = {num_edges} * {cable_len} * {cable_cost} = {total_cable_cost:.2f}")

    total = total_port_cost + total_cable_cost
    lines.append(f"5) Total cost = {total:.2f}")

    return "\n".join(lines), total_port_cost, total_cable_cost, total


# ============================================================================================
# Sidebar Inputs
# ============================================================================================
st.sidebar.title("Controls")

with st.sidebar.form("inputs"):
    st.sidebar.markdown("### Design Parameters")
    nodes = st.number_input("Number of Nodes", min_value=1, value=6)
    topology = st.selectbox("Topology", ["Bus", "Star", "Ring", "Mesh", "Tree"])
    ring_variant = st.selectbox("Ring Variant", ["Singly (Bidirectional)", "Singly (Unidirectional)", "Doubly (Unidirectional)"])

    st.sidebar.markdown("### Cost Parameters")
    port_cost = st.number_input("Cost per Port (₹)", min_value=0.0, value=100.0)
    cable_len = st.number_input("Cable Length per Connection (m)", min_value=0.0, value=10.0)
    cable_cost = st.number_input("Cable Cost per Meter (₹)", min_value=0.0, value=50.0)

    generate = st.form_submit_button("Generate Topology")


# ============================================================================================
# Wider Layout: Simulator gets 75% width
# ============================================================================================
col1, col2 = st.columns([3, 1])      # 🟢 Wider simulator area


# ============================================================================================
# MAIN SIMULATOR AREA
# ============================================================================================
with col1:
    st.header("Network Topology Simulator 🛠")

    if not generate:
        st.info("Click **Generate Topology** to continue.")
        st.stop()

    # ---------------- BUILD GRAPH ----------------
    if topology == "Ring" and ring_variant in ["Singly (Unidirectional)", "Doubly (Unidirectional)"]:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(1, int(nodes) + 1))

    if topology == "Bus":
        for i in range(1, int(nodes)):
            G.add_edge(i, i + 1)

    elif topology == "Star":
        for i in range(2, int(nodes) + 1):
            G.add_edge(1, i)

    elif topology == "Ring":
        for i in range(1, int(nodes)):
            G.add_edge(i, i + 1)
        G.add_edge(int(nodes), 1)

        if ring_variant == "Doubly (Unidirectional)":
            for i in range(1, int(nodes)):
                G.add_edge(i + 1, i)
            G.add_edge(1, int(nodes))

    elif topology == "Mesh":
        n = int(nodes)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                G.add_edge(i, j)

    elif topology == "Tree":
        n = int(nodes)
        if not math.log2(n + 1).is_integer():
            st.error("Tree requires nodes = 2^k - 1 (3, 7, 15, 31...)")
            st.stop()
        for i in range(2, n + 1):
            G.add_edge(i // 2, i)

    # ---------------- COST CALC ----------------
    num_edges = G.number_of_edges()
    step_text, port_total, cable_total, total_cost = build_step_by_step(
        int(nodes), num_edges, port_cost, cable_len, cable_cost
    )

    st.subheader("Cost Breakdown")
    st.text_area("Summary & Steps", step_text, height=230)

    # ---------------- NODE POSITIONS ----------------
    if topology == "Bus":
        xs = np.arange(len(G.nodes()))
        ys = np.array([0.6 if i % 2 == 0 else -0.6 for i in range(len(G.nodes()))])
        pos = {node: (float(x), float(y)) for node, x, y in zip(G.nodes(), xs, ys)}

    elif topology == "Ring":
        theta = np.linspace(0, 2 * np.pi, len(G.nodes()), endpoint=False)
        pos = {node: (np.cos(th), np.sin(th)) for node, th in zip(G.nodes(), theta)}

    elif topology == "Tree":
        pos = hierarchy_pos(G, 1)

    else:
        pos = nx.spring_layout(G, seed=42)

    # ---------------- DRAW GRAPH ----------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_axis_off()

    if topology != "Bus":
        edges_drawn = set()
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            if topology == "Ring" and ring_variant == "Doubly (Unidirectional)":
                pair = tuple(sorted((u, v)))
                if pair not in edges_drawn:
                    draw_double_edge_with_arrows(ax, x1, y1, x2, y2)
                    edges_drawn.add(pair)
            else:
                ax.plot([x1, x2], [y1, y2], color="#333", linewidth=2)

                if topology == "Ring" and ring_variant == "Singly (Unidirectional)":
                    dx, dy = x2 - x1, y2 - y1
                    ax.add_patch(FancyArrow((x1 + x2) / 2, (y1 + y2) / 2,
                                            dx * 0.0001, dy * 0.0001,
                                            width=0.008, head_width=0.05, head_length=0.05))

    else:
        xs = np.arange(len(G.nodes()))
        left, right = -0.5, len(G.nodes()) - 0.5
        ax.plot([left, right], [0, 0], color="black", linewidth=4)
        for tx in [left, right]:
            ax.add_patch(Rectangle((tx - 0.08, -0.12), 0.16, 0.24, color="black"))
        for node, (x, y) in pos.items():
            ax.plot([x, x], [y, 0], color="black")
            ax.plot([x - 0.06, x + 0.06], [0, 0], color="black")

    # ---------------- DRAW NODES ----------------
    if computer_img:
        img_arr = np.array(computer_img)
        zoom = 0.12 if len(pos) <= 8 else (0.09 if len(pos) <= 16 else 0.06)
        for node, (x, y) in pos.items():
            ab = AnnotationBbox(OffsetImage(img_arr, zoom=zoom),
                                (x, y), frameon=False)
            ax.add_artist(ab)
            ax.text(x, y + 0.02, str(node),
                    ha="center", color="white", fontsize=12, fontweight="bold")
    else:
        nx.draw_networkx_nodes(G, pos, node_size=900)
        nx.draw_networkx_labels(G, pos)

    st.subheader("Network Graph")
    st.pyplot(fig)

    # ---------------- DOCX REPORT ----------------
    def generate_docx():
        doc = Document()
        doc.add_heading("Network Topology Report", 0)

        doc.add_heading("Details", level=1)
        tbl = doc.add_table(rows=7, cols=2)
        data = [
            ("Topology", topology),
            ("Ring Variant", ring_variant if topology == "Ring" else "N/A"),
            ("Nodes", str(nodes)),
            ("Cost per Port", str(port_cost)),
            ("Cable Length (m)", str(cable_len)),
            ("Cable Cost (₹/m)", str(cable_cost)),
            ("Total Cost", f"₹{total_cost:.2f}")
        ]
        for r, (k, v) in enumerate(data):
            tbl.rows[r].cells[0].text = k
            tbl.rows[r].cells[1].text = v

        doc.add_heading("Cost Breakdown", level=1)
        doc.add_paragraph(step_text)

        doc.add_heading("Network Diagram", level=1)
        img = BytesIO()
        fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
        img.seek(0)
        doc.add_picture(img, width=Inches(6))

        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    st.download_button(
        "Download Report (.docx)",
        data=generate_docx(),
        file_name="network_topology_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

# ============================================================================================
# RIGHT SIDEBAR — COLLAPSIBLE ABOUT / DEVELOPERS / GUIDE
# ============================================================================================
with col2:

    with st.expander("📌 About This Project"):
        st.write(
            "This simulator visualizes common network topologies and computes "
            "the total setup cost based on ports and cable usage."
        )
        st.write("Reference: https://www.youtube.com/watch?v=zbqrNg4C98U")

    with st.expander("👨‍💻 Developers"):
        c1, c2 = st.columns(2)
        with c1:
            if abijith_img:
                st.image(abijith_img, width=140, caption="Abijith Thennarasu")
        with c2:
            if dharmayu_img:
                st.image(dharmayu_img, width=140, caption="Dharmayu Jadwani")

    with st.expander("🎓 Guide / Faculty"):
        if prof_img:
            st.image(prof_img, width=180, caption="Dr. Swaminathan Annadurai")
