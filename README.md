GA4GH-RegBot: Compliance Assistant
Status: Proposal Stage for GSoC 2026

Overview
RegBot is an LLM-powered tool designed to help researchers map their consent forms against GA4GH regulatory frameworks. It uses RAG (Retrieval-Augmented Generation) to flag compliance gaps automatically.

What works today
- **Ingest** policy PDFs or `.txt` files into a local **Chroma** store plus a JSON manifest (chunk ids, page hints, source metadata).
- **Hybrid retrieval**: embedding search + **BM25**, merged with reciprocal rank fusion.
- **Compliance pass**: one OpenAI JSON call when `OPENAI_API_KEY` is set; otherwise a small keyword gap heuristic that still returns chunk citations.
- **Streamlit UI** for upload + paste flows (`src/streamlit_app.py`).
- **CLI**: `python -m src.main …` (see below).
- **Citation grounding (programmatic):** LLM JSON is checked so every `citations[].chunk_id` is in the retrieved evidence set; by default citations must be at least as numerous as recommendations. Failed checks trigger **one automatic rewrite request** to the model with the allow-list of ids.
- **PDF eval harness:** `eval` subcommand ingests a real GA4GH PDF and prints retrieval hits for built-in or custom queries (for manual review / building a gold set later).

Quickstart (Development)
- Prerequisites: **Python 3.10–3.12** (CI uses 3.11). Python 3.14 is not supported yet for the full stack (native wheels for parts of the ML/Chroma toolchain often lag).
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

- Ingest a policy file into `./data/regbot_store` (use `--reset` when reloading the same corpus):

```bash
python -m src.main ingest --path path/to/policy.pdf --reset
```

- Check a consent / data-use text file:

```bash
python -m src.main check --consent path/to/consent.txt
```

- Run the Streamlit UI from the repo root:

```bash
python -m streamlit run src/streamlit_app.py
```

- End-to-end sample (synthetic policy + consent under `examples/`):

```bash
python examples/run_demo.py
```

Evaluate retrieval on a **real** GA4GH PDF (resets the store by default if you pass `--reset`):

```bash
python -m src.main eval --pdf path/to/ga4gh_policy.pdf --reset --top-k 8
```

Use your own query list (one line per query):

```bash
python -m src.main eval --pdf path/to/ga4gh_policy.pdf --reset --queries-file examples/eval/queries_ga4gh.txt
```

Optionally append a full compliance JSON report for a consent file:

```bash
python -m src.main eval --pdf path/to/ga4gh_policy.pdf --reset --consent path/to/consent.txt
```

Run tests

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

Environment Variables
- `OPENAI_API_KEY`: Optional; enables the JSON LLM compliance pass via `REGBOT_LLM_MODEL` (default `gpt-4o-mini`).
- `REGBOT_STORE`: Optional override for the on-disk store directory (default `./data/regbot_store`).
- `REGBOT_EMBEDDING_MODEL`: Optional SentenceTransformers model id (default `sentence-transformers/all-MiniLM-L6-v2`).

Architecture (implemented vs planned)
- **Core:** Python 3, modular package under `src/regbot/` (ingest, hybrid retrieval, compliance).
- **Embeddings:** `sentence-transformers` (default `all-MiniLM-L6-v2`).
- **Vector store:** Chroma persistent store under `REGBOT_STORE/chroma` plus `manifest.json` for BM25 text.
- **Retrieval:** cosine similarity in Chroma + `rank-bm25`, fused via reciprocal rank fusion; optional metadata category filter.
- **LLM:** OpenAI Chat Completions JSON mode when `OPENAI_API_KEY` is set; offline keyword-style fallback otherwise.
- **UI:** Streamlit (`src/streamlit_app.py`).
- **Also listed for roadmap:** deeper LangChain/LlamaIndex integration, richer evaluation, and stronger citation QA.

Roadmap
Phase 1: Ingest GA4GH "Framework for Responsible Sharing" policy documents.

Phase 2: Build RAG pipeline for clause extraction.

Phase 3: Develop Streamlit frontend for user uploads.
