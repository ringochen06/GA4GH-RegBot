# Demo (local)

From the repository root, with a virtualenv activated and dependencies installed:

1. Ingest the bundled synthetic policy text (resets the local store):

```bash
python -m src.main --store ./data/regbot_store ingest --path examples/data/sample_ga4gh_policy_stub.txt --reset
```

2. Run a check against the sample consent:

```bash
python -m src.main --store ./data/regbot_store check --consent examples/data/sample_consent_short.txt
```

3. Optional UI:

```bash
python -m streamlit run src/streamlit_app.py
```

Set `OPENAI_API_KEY` in your environment for JSON output from the configured chat model (`REGBOT_LLM_MODEL`, default `gpt-4o-mini`). Without a key, the tool still retrieves policy chunks and returns a small keyword-style gap summary.
