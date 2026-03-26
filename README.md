GA4GH-RegBot: Compliance Assistant
Status: Proposal Stage for GSoC 2026

Overview
RegBot is an LLM-powered tool designed to help researchers map their consent forms against GA4GH regulatory frameworks. It uses RAG (Retrieval-Augmented Generation) to flag compliance gaps automatically.

Quickstart (Development)
- Prerequisites: Python 3.10+ recommended (Python 3.11 is used in CI)
- Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- Configure environment variables:
  - Export variables in your shell (recommended)
  - If you use a local `.env`, keep it private and do not commit it

- Run the current placeholder pipeline:

```bash
python src/main.py
```

Run tests

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

Environment Variables
- `OPENAI_API_KEY`: Optional for future LLM integration. The current code will read it if present.

Architecture (Planned)
Core: Python

LLM Framework: LangChain / LlamaIndex

Vector Store: ChromaDB / FAISS

UI: Streamlit

Roadmap
Phase 1: Ingest GA4GH "Framework for Responsible Sharing" policy documents.

Phase 2: Build RAG pipeline for clause extraction.

Phase 3: Develop Streamlit frontend for user uploads.
