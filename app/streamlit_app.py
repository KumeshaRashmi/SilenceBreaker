"""SilenceBreaker demo UI (Streamlit).

Run:
    python -m src.rag.build_index        # build the index first
    streamlit run app/streamlit_app.py
"""
import sys
import os
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Must be set before any torch/transformers imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
from src import config

st.set_page_config(page_title="SilenceBreaker", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Always-visible crisis banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background:#b71c1c;color:white;padding:10px 16px;border-radius:6px;margin-bottom:12px;font-size:0.9rem;">
    🚨 <strong>If you are in immediate danger, call emergency services now.</strong>
    &nbsp;|&nbsp; UK: <strong>999</strong>
    &nbsp;|&nbsp; US/Canada: <strong>911</strong>
    &nbsp;|&nbsp; AU: <strong>000</strong>
    &nbsp;|&nbsp; EU: <strong>112</strong>
    &nbsp;|&nbsp; Sri Lanka: <strong>119</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ SilenceBreaker — Support Assistant (Prototype)")
st.caption("General information only. Not legal, medical, or emergency advice. "
           "This prototype does not store your data.")

if config.OFFLINE:
    st.info("Running in OFFLINE mode (no LLM API key set). Responses use a "
            "deterministic template. Add a key in .env for richer answers.")

# ---------------------------------------------------------------------------
# Session state for chat history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def run_pipeline(text: str) -> dict:
    """Run the pipeline in a subprocess to avoid PyTorch/Streamlit conflicts on Windows."""
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "src.graph", text],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Pipeline process exited with an error.")
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(
        f"No JSON output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
txt = st.text_area(
    "Describe what's happening (you can stay anonymous):",
    height=140,
    placeholder="e.g. My manager keeps messaging me late at night and hints I'll "
                "be fired if I complain...",
)

col_btn, col_clear, _ = st.columns([1, 1, 3])
go = col_btn.button("Get support", type="primary")
if col_clear.button("Clear history"):
    st.session_state.history = []
    st.rerun()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if go and txt.strip():
    with st.spinner("Analysing..."):
        try:
            r = run_pipeline(txt)
        except FileNotFoundError as exc:
            st.error(f"{exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

    st.session_state.history.append({"text": txt, "result": r})

elif go:
    st.warning("Please enter a description first.")

# ---------------------------------------------------------------------------
# Display history (most recent first)
# ---------------------------------------------------------------------------
for entry in reversed(st.session_state.history):
    r = entry["result"]
    st.divider()

    if r["urgent"]:
        st.error("⚠️ Urgent situation detected. If you are in immediate danger, "
                 "contact your local emergency number or a crisis helpline now.")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Category", r["category"].replace("_", " ").title())
    c2.metric("Distress", r["distress"].title())
    c3.metric("Risk", r["risk"].title())

    # Confidence scores (shown when available)
    if r.get("emotion_score") is not None:
        c4.metric("Emotion confidence", f"{r['emotion_score']:.0%}")
    else:
        c4.metric("Emotion", r.get("emotion", "—").title())

    if r.get("is_abuse") is not None:
        label = "Flagged" if r["is_abuse"] else "Not flagged"
        conf = r.get("abuse_conf")
        c5.metric("Toxicity (exp.)", label,
                  delta=f"{conf:.0%} conf" if conf else None,
                  delta_color="off")
    else:
        c5.metric("Toxicity (exp.)", "N/A")

    st.markdown(f"**Your message:** *{entry['text']}*")
    st.subheader("Guidance")
    st.write(r["response"])

    with st.expander("📑 Evidence used (retrieved sources)"):
        for i, e in enumerate(r["evidence"], 1):
            st.markdown(f"**[{i}] {e['source']}** · similarity={e['score']:.2f}")
            st.write(e["text"])

st.divider()
st.caption("SilenceBreaker is an academic prototype. It is not affiliated with any "
           "support organisation and must not be relied on in an emergency.")
