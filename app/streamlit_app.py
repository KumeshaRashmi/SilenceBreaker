"""SilenceBreaker demo UI (Streamlit).

Run:
    python -m src.rag.build_index        # build the index first
    streamlit run app/streamlit_app.py
"""
import sys
import os

# Make `src` importable when run via `streamlit run app/streamlit_app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.graph import run
from src import config

st.set_page_config(page_title="SilenceBreaker", page_icon="🛡️", layout="wide")

st.title("🛡️ SilenceBreaker — Support Assistant (Prototype)")
st.caption("General information only. Not legal, medical, or emergency advice. "
           "This prototype does not store your data.")

if config.OFFLINE:
    st.info("Running in OFFLINE mode (no LLM API key set). Responses use a "
            "deterministic template. Add a key in .env for richer answers.")

txt = st.text_area(
    "Describe what's happening (you can stay anonymous):",
    height=140,
    placeholder="e.g. My manager keeps messaging me late at night and hints I'll "
                "be fired if I complain...",
)

col_btn, _ = st.columns([1, 4])
go = col_btn.button("Get support", type="primary")

if go and txt.strip():
    with st.spinner("Analysing..."):
        try:
            r = run(txt)
        except FileNotFoundError as exc:
            st.error(f"{exc}")
            st.stop()

    if r["urgent"]:
        st.error("⚠️ This may be an urgent situation. If you are in immediate "
                 "danger, contact your local emergency number or a crisis "
                 "helpline now.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Category", r["category"].replace("_", " ").title())
    c2.metric("Distress", r["distress"].title())
    c3.metric("Risk", r["risk"].title())

    st.subheader("Guidance")
    st.write(r["response"])

    with st.expander("📑 Evidence used (retrieved sources)"):
        for i, e in enumerate(r["evidence"], 1):
            st.markdown(f"**[{i}] {e['source']}** · similarity={e['score']:.2f}")
            st.write(e["text"])

    st.divider()
    st.caption("This prototype is not a substitute for professional or "
               "emergency help.")
elif go:
    st.warning("Please enter a description first.")
