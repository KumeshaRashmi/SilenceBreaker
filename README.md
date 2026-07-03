# SilenceBreaker

**A Multi-Agent, Risk-Aware, RAG-Based Support Assistant for Abuse-Related Help-Seeking**

Advanced AI – Group Project. SilenceBreaker is a **prototype** that identifies
abuse-related concerns from text, retrieves verified support information, and
generates **evidence-grounded** guidance with **risk-escalation flags**.

## Team

| Index No. | Name |
|---|---|
| EG/2021/4685 | Muthukumari H.M.S. |
| EG/2021/4738 | Ranasinghe R.M.W.O. |
| EG/2021/4748 | Rashmi E.D.K. |
| EG/2021/4775 | Samarasinghe C.Y. |

> **Scope & ethics.** This is a prototype information/triage tool. It does **not**
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

### AI techniques used
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
> rule-based category classifier and a template planner, so the whole pipeline is
> deterministic and reproducible. Set an LLM key in `.env` for the full generative experience.

---

## Datasets

| Purpose | Dataset | Link |
|---|---|---|
| Emotion / distress | `dair-ai/emotion` | https://huggingface.co/datasets/dair-ai/emotion |
| Abuse classifier training | `cardiffnlp/tweet_eval` (hate subtask) | https://huggingface.co/datasets/cardiffnlp/tweet_eval |
| Abuse classifier cross-domain eval | Jigsaw Toxic Comment | https://huggingface.co/datasets/google/jigsaw_toxicity_pred |
| Ready emotion model | `j-hartmann/emotion-english-distilroberta-base` | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base |
| Knowledge base | curated in `data/kb/` | original content, cited per-document |

The 4-way abuse category (domestic / workplace / coercive / non-abuse) has no clean
public dataset, so it is handled by a **few-shot LLM** (in-context learning) and/or a
small hand-labelled set. The binary abuse/hate-speech detector is fine-tuned on
`cardiffnlp/tweet_eval` (hate subtask) and evaluated two ways: in-distribution on its
own test split, and cross-domain on Jigsaw (Wikipedia comments) to see how well a
Twitter-trained classifier generalises to a different register of text.

---

## Evaluation

```bash
python -m evaluation.eval_classifier      # Accuracy / Precision / Recall / F1
python -m evaluation.eval_retrieval       # hit@k, precision@k
python -m evaluation.eval_faithfulness    # category & risk accuracy + faithfulness
python -m evaluation.ablation             # A (LLM) vs B (RAG) vs C (multi-agent)
```

The **ablation study** (baseline vs RAG vs full multi-agent) is the most informative
evaluation for understanding what each architectural layer contributes. Edit
`evaluation/test_prompts.py` to extend the test set.

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
disclaimers and verified contact details (helplines, support organisations, legal
aid) for the UK, US, Canada, Australia, and Sri Lanka, sourced from official
organisations and cited at the bottom of each document. Do not invent contact
details or statistics if you extend this knowledge base.

---
