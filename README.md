# SilenceBreaker 🛡️

**A Multi-Agent, Risk-Aware, RAG-Based Support Assistant for Abuse-Related Help-Seeking**

Advanced AI – Group Project 2026. SilenceBreaker is a **prototype** that identifies
abuse-related concerns from text, retrieves verified support information, and
generates **evidence-grounded** guidance with **risk-escalation flags**.

> ⚠️ **Scope & ethics.** This is a prototype information/triage tool. It does **not**
> provide legal, medical, or professional advice, does **not** diagnose, and routes
> high-risk cases to *seek immediate local help*. It is not a substitute for
> professional or emergency services.

---

## What it does

```
User text ─▶ Agent 1 Listener ─▶ Agent 2 Retriever ─▶ Agent 4 Risk ─▶ Agent 3 Planner ─▶ Response
            (category + distress)   (RAG evidence)      (low/med/high)   (grounded reply)
```

- **Agent 1 – Situation Listener:** abuse category (few-shot LLM) + distress level
  (emotion transformer) + abusive/non-abusive (fine-tuned DistilBERT, optional).
- **Agent 2 – RAG Retriever:** MiniLM embeddings + FAISS over a curated knowledge base.
- **Agent 4 – Risk Evaluator:** distress + danger-cue rules → low / medium / high.
- **Agent 3 – Safety Planner:** LLM grounded only on retrieved evidence; urgent help first when risk is high.
- Orchestrated with **LangGraph** (falls back to a sequential pipeline if not installed).

### AI techniques demonstrated (assignment requires ≥3; this uses 5)
NLP preprocessing & classification · Transformers (DistilBERT/RoBERTa fine-tuning) ·
Embeddings + RAG · Prompt engineering · Few-shot / in-context learning.

---

## Quick start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) LLM key for higher-quality responses
cp .env.example .env        # then add a Groq/OpenAI key. Leave blank for OFFLINE mode.

# 3. Build the retrieval index from data/kb/
python -m src.rag.build_index

# 4. (Optional but recommended) train the binary abuse detector
python train/train_abuse_clf.py

# 5. Test the pipeline from the command line
python -m src.graph

# 6. Run the demo UI
streamlit run app/streamlit_app.py
#    or:  ./run_demo.sh
```

> **OFFLINE mode:** with no API key set, the project still runs end-to-end using a
> rule-based category classifier and a template planner. This makes grading
> reproducible. Set an LLM key in `.env` for the full generative experience.

---

## Datasets (from assignment-approved sources)

| Purpose | Dataset | Link |
|---|---|---|
| Emotion / distress | `dair-ai/emotion` | https://huggingface.co/datasets/dair-ai/emotion |
| Abuse / harassment text | Jigsaw Toxic Comment | https://huggingface.co/datasets/google/jigsaw_toxicity_pred |
| Ready emotion model | `j-hartmann/emotion-english-distilroberta-base` | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base |
| Knowledge base | curated in `data/kb/` | (write/verify + cite your own sources) |

The 4-way abuse category (domestic / workplace / coercive / non-abuse) has no clean
public dataset, so it is handled by a **few-shot LLM** (in-context learning) and/or a
small hand-labelled set. The binary abuse detector is trained on Jigsaw for
quantitative rigor.

---

## Evaluation

```bash
python -m evaluation.eval_classifier      # Accuracy / Precision / Recall / F1
python -m evaluation.eval_retrieval       # hit@k, precision@k
python -m evaluation.eval_faithfulness    # category & risk accuracy + faithfulness
python -m evaluation.ablation             # A (LLM) vs B (RAG) vs C (multi-agent)
```

The **ablation study** (baseline vs RAG vs full multi-agent) is the key evaluation
for marks. Edit `evaluation/test_prompts.py` to extend the test set.

---

## Project structure

```
silencebreaker/
├── src/                # config, preprocess, llm, graph
│   ├── agents/         # listener, retriever, risk, planner
│   └── rag/            # build_index
├── train/              # train_abuse_clf.py, train_emotion_clf.py
├── evaluation/         # eval_* + ablation + test_prompts
├── data/kb/            # curated knowledge base (RAG source)
├── notebooks/          # documented walkthrough
└── app/                # streamlit_app.py
```

---

## Knowledge base note

The documents in `data/kb/` are **general, original information** with clear
disclaimers. Several contain `ACTION FOR THE TEAM` markers where you must insert
**verified** national/regional helpline details and **cite** the official source in
your report. Do not invent contact details or statistics.

---

## Disclaimer

SilenceBreaker is an academic prototype. It is not affiliated with any support
organisation and must not be relied on in an emergency. Always contact local
emergency services or a qualified professional for real situations.
