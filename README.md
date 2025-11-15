# Network Topology Simulator (Streamlit Web App)

A fully interactive web-based network topology visualization and cost estimation tool.
Supports:
- Bus Topology
- Star Topology
- Ring Topology (with unidirectional / bidirectional variants)
- Mesh Topology
- Tree Topology (2^k - 1 nodes)

Features:
- Graph drawing using NetworkX + Matplotlib
- Cost calculations (port + cable cost)
- Step-by-step explanation
- DOCX report generator with graph image
- Image support for node icons & developer/guide photos

---

## 🚀 Live Demo (if deployed)
(Add your Streamlit Cloud / Render URL here)

---

## 🏗 Folder Structure

network-topology-simulator/
│
├── app.py
├── requirements.txt
├── README.md
│
└── assets/
├── computer.png
├── professor.png
├── abijith.jpeg
├── dharmyu.jpeg


---

## ▶ Running Locally

### 1. Install Python
Make sure Python 3.8+ is installed.

### 2. Install dependencies
```bash
pip install -r requirements.txt
