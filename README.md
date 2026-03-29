GA4GH-RegBot: Compliance Assistant
Status: **MVP available** — ingest, hybrid retrieval, optional LLM compliance + programmatic citation checks, CLI, Streamlit, and a small PDF eval harness. Ongoing work: real-corpus evaluation, stricter schemas, and contributor tooling.

Overview
RegBot is an LLM-powered tool designed to help researchers map their consent forms against GA4GH regulatory frameworks. It uses RAG (Retrieval-Augmented Generation) to flag compliance gaps automatically.

What works today
- **Ingest** policy PDFs or `.txt` files into a local **Chroma** store plus a JSON manifest (chunk ids, page hints, source metadata).
- **Hybrid retrieval**: embedding search + **BM25**, merged with reciprocal rank fusion.
- **Compliance pass**: one OpenAI JSON call when `OPENAI_API_KEY` is set; otherwise a small keyword gap heuristic that still returns chunk citations.
- **Streamlit UI** for upload + paste flows (`src/streamlit_app.py`).
- **CLI**: `python -m src.main …` (see below).
- **Citation grounding (programmatic):** Each `recommendations[]` item must be `{ "text": "...", "evidence_chunk_ids": ["..."] }` with ids taken **only** from retrieved chunks; optional `citations[]` must also respect the same allow-list. Failed checks trigger **one automatic rewrite request** with the allow-list.
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
- `REGBOT_MIN_TOKEN_OVERLAP`: For the LLM path, minimum **token recall** between each recommendation and the cited chunk texts (default `0.06`). Set to `0` to disable dropping rows for low overlap (scores may still be attached).
- `REGBOT_CHROMA_ANONYMIZED_TELEMETRY`: Set to `1` to enable Chroma client telemetry; default is off (`0`).
- `REGBOT_OPENAI_MAX_RETRIES`: Maximum retries for transient OpenAI API errors (default `3`).

Architecture (implemented vs planned)
- **Core:** Python 3, modular package under `src/regbot/` (ingest, hybrid retrieval, compliance).
- **Embeddings:** `sentence-transformers` (default `all-MiniLM-L6-v2`).
- **Vector store:** Chroma persistent store under `REGBOT_STORE/chroma` plus `manifest.json` for BM25 text.
- **Retrieval:** cosine similarity in Chroma + `rank-bm25`, fused via reciprocal rank fusion; optional metadata category filter.
- **LLM:** OpenAI Chat Completions JSON mode when `OPENAI_API_KEY` is set; offline keyword-style fallback otherwise.
- **UI:** Streamlit (`src/streamlit_app.py`).
- **Optional / roadmap:** optional LangChain/LlamaIndex adapters on top of the same stores; richer offline evaluation (Ragas, human labels); structured per-recommendation evidence fields.

Next steps (suggested priorities)
1. **Real GA4GH corpus**: ingest official PDFs, tune chunk size/overlap and hybrid fusion weights using `eval` + a small **gold query → chunk_id** list (manual or semi-automated).
2. **Stricter outputs:** `evidence_chunk_ids[]` plus programmatic ID checks, token-overlap filtering on the LLM path (`REGBOT_MIN_TOKEN_OVERLAP`), and retries when grounding/overlap fails. **Next:** richer evidence objects (e.g. optional quotes), stricter refusal when excerpts are insufficient.
3. **Contributor experience**: **Done in-repo:** separate **Lint** workflow (Ruff check + format check), `CONTRIBUTING.md`, `.pre-commit-config.yaml`, `pyproject.toml`, `requirements-dev.txt`. **Still open:** optional CI `mypy`, broader type hints, Black-only rules if the team wants them.
4. **Operational hardening**: **Done in-repo:** Chroma telemetry off by default (`REGBOT_CHROMA_ANONYMIZED_TELEMETRY`), OpenAI client `max_retries` via `REGBOT_OPENAI_MAX_RETRIES`, clear `ValueError` when a PDF yields no extractable text. **Next:** optional request timeouts, Chroma/OpenAI observability hooks.

Contributing
- See **`CONTRIBUTING.md`** for venv setup, **Ruff** lint/format, optional **pre-commit**, and tests.
- Open PRs against the upstream repo; keep changes scoped and tested (`python -m unittest discover -s tests -p "test*.py" -v`). Do not commit `.env`, API keys, or local `data/regbot_store/`.
