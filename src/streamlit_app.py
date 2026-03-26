from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.main import RegBot

st.set_page_config(page_title="GA4GH-RegBot", layout="wide")
load_dotenv()

st.title("GA4GH-RegBot")
st.caption(
    "Prototype assistant: ingest GA4GH-style policy excerpts, retrieve hybrid context, "
    "and draft a citation-oriented compliance note. Not legal advice."
)

with st.sidebar:
    store_dir = st.text_input(
        "Store directory",
        value=os.getenv("REGBOT_STORE", "./data/regbot_store"),
        help="Where Chroma + manifest.json are written.",
    )
    st.markdown("Set `OPENAI_API_KEY` locally for full LLM JSON analysis; otherwise a heuristic fallback runs.")

tab_ingest, tab_check = st.tabs(["Ingest policy", "Check consent"])

with tab_ingest:
    uploaded = st.file_uploader("Policy PDF or .txt", type=["pdf", "txt"])
    reset = st.checkbox("Reset store before ingest", value=False)
    category = st.text_input("Category label (optional)", value="")
    if st.button("Ingest", type="primary") and uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            bot = RegBot(store_dir=store_dir)
            ok = bot.ingest_policy_documents(
                tmp_path,
                reset=reset,
                category=category.strip() or None,
            )
            st.success("Ingest finished." if ok else "Ingest reported a problem (see terminal/logs).")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

with tab_check:
    consent_text = st.text_area("Paste consent or data-use language", height=240)
    filter_cat = st.text_input("Only use policy category (optional)", value="")
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=16, value=8)
    if st.button("Analyze", type="primary"):
        bot = RegBot(store_dir=store_dir)
        report = bot.check_compliance(
            consent_text,
            category=filter_cat.strip() or None,
            top_k=int(top_k),
        )
        st.subheader("Report")
        st.json(report)
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="regbot_report.json",
            mime="application/json",
        )
