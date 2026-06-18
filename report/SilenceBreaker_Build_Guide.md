# SilenceBreaker — Complete Build Guide

**A Multi-Agent, Risk-Aware, RAG-Based Support Assistant for Abuse-Related Help-Seeking**

This is a full, reproducible, step-by-step guide for the *Advanced AI – Group Project 2026*. It includes the project structure, dataset links, **all the code**, evaluation scripts, the report outline mapped to the marking rubric, and a 4-member / 4-week plan.

> **Read this first — scope and ethics.** SilenceBreaker is a **prototype triage and information-retrieval tool**. It does **not** give legal/medical/professional advice, does **not** diagnose, and **must** route high-risk cases to "seek immediate local help." Keep this framing everywhere — it is worth real marks (Problem Relevance + Discussion/Ethics) and it is the responsible thing to do.

---

## 0. How this maps to your marking rubric (100 pts)

| Criterion | Pts | How SilenceBreaker earns it |
|---|---|---|
| Problem Relevance & Scope | 10 | Narrow, well-justified domain (domestic abuse + workplace harassment, English, text-only). |
| Use of AI Techniques (≥3) | 20 | You use **5**: NLP preprocessing+classification, Transformers (DistilBERT/RoBERTa fine-tune), Embeddings/RAG, Prompt engineering, Few-shot/in-context learning. |
| Model Implementation & Creativity | 20 | 4-agent LangGraph pipeline, RAG grounding, risk-aware generation. |
| Evaluation & Metrics | 15 | Classification (Acc/P/R/F1), Retrieval (P@k, hit-rate), Faithfulness %, **baseline vs RAG vs multi-agent comparison**. |
| Report Quality | 15 | Use the exact section structure in §11. APA refs, diagrams, tables. |
| Working Demo | 15 | Streamlit app with category/distress/risk badges, evidence panel, disclaimer. |
| Team & Presentation | 5 | 4 clear technical roles + contribution table (§12). |

**The single biggest mark-booster:** the ablation study — *baseline chatbot vs RAG chatbot vs full multi-agent system* (§9). Do not skip it.

---

## 1. Tech stack

```
Python 3.10+
PyTorch + Hugging Face Transformers   # classifiers + optional local LLM
sentence-transformers + FAISS         # embeddings + retrieval (RAG)
LangGraph + LangChain                 # multi-agent orchestration
Streamlit                             # demo UI
scikit-learn, pandas, numpy           # metrics + data
datasets                              # load HF datasets
```

**Model choices (free / lightweight):**
- Classifier base: `distilbert-base-uncased` (fast) or `roberta-base` (stronger).
- Emotion model: `j-hartmann/emotion-english-distilroberta-base` (ready-made, no training needed) **or** fine-tune your own on `dair-ai/emotion`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- LLM (Agent 3): any OpenAI-compatible endpoint. **Free option:** Groq (`llama-3.1-8b-instant`) — fast and free tier. Alternatives: OpenAI, Together, or a **local** HF instruct model if you have a GPU.

---

## 2. Project structure

```
silencebreaker/
├── README.md
├── requirements.txt
├── .env                      # API keys (never commit)
├── data/
│   ├── raw/
│   ├── processed/
│   └── kb/                   # knowledge base .md/.txt docs (20–40)
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── agents/
│   │   ├── listener.py       # Agent 1: classify + distress
│   │   ├── retriever.py      # Agent 2: RAG
│   │   ├── planner.py        # Agent 3: grounded response
│   │   └── risk.py           # Agent 4: risk escalation
│   ├── rag/
│   │   ├── build_index.py
│   │   └── store.py
│   ├── llm.py                # LLM wrapper
│   └── graph.py              # LangGraph orchestration
├── train/
│   ├── train_abuse_clf.py
│   └── train_emotion_clf.py  # optional (or use ready model)
├── evaluation/
│   ├── eval_classifier.py
│   ├── eval_retrieval.py
│   ├── eval_faithfulness.py
│   └── ablation.py
└── app/
    └── streamlit_app.py
```

`requirements.txt`:
```
torch
transformers
datasets
sentence-transformers
faiss-cpu
scikit-learn
pandas
numpy
langchain
langgraph
langchain-community
streamlit
python-dotenv
openai
matplotlib
```

Setup:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`.env`:
```
GROQ_API_KEY=your_key_here
# or OPENAI_API_KEY=...  /  TOGETHER_API_KEY=...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

---

## 3. Datasets (with links)

All are from the sources your assignment explicitly allows (Kaggle / Hugging Face / UCI / OpenML).

| Purpose | Dataset | Link | Notes |
|---|---|---|---|
| **Distress / emotion** (Agent 1 + Agent 4) | `dair-ai/emotion` | https://huggingface.co/datasets/dair-ai/emotion | 6 emotions; map to low/med/high distress. Loads in one line. |
| **Abuse vs non-abuse** (Agent 1 binary head) | Jigsaw Toxic Comment | https://huggingface.co/datasets/google/jigsaw_toxicity_pred  •  https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge | Use `toxic`/`threat`/`insult` as proxy for abusive/harassing text. |
| **Ready emotion model** (skip training) | `j-hartmann/emotion-english-distilroberta-base` | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base | Drop-in classifier; cite it. |
| **RAG knowledge base** | You curate it | — | 20–40 short docs you write/collect (see §6). This is the *trusted* source for grounding. |

> **Honest note on the 4-way abuse category** (domestic / workplace / coercive control / non-abuse): there is **no clean public dataset** with exactly these labels. Two legitimate options — pick one and document it:
> 1. **Few-shot LLM classifier** (recommended, fast): zero/few-shot prompt the LLM to assign the category. Counts as *in-context learning* (a required technique). Code in §5.
> 2. **Small curated dataset**: hand-label ~150–300 short scenarios (or generate with an LLM and manually verify) into the 4 classes, then fine-tune DistilBERT. Document your labeling methodology — the assignment requires this for custom data.
>
> A strong submission does **both**: train a binary abuse detector on Jigsaw (quantitative rigor) **and** use the few-shot LLM for the fine-grained category (creativity + in-context learning). That's exactly what examiners reward.

Quick load test:
```python
from datasets import load_dataset
emo = load_dataset("dair-ai/emotion")
print(emo)                       # train/validation/test
print(emo["train"][0])           # {'text': ..., 'label': 0}
print(emo["train"].features["label"].names)
# ['sadness','joy','love','anger','fear','surprise']
```

---

## 4. Config + preprocessing

`src/config.py`:
```python
import os
from dotenv import load_dotenv
load_dotenv()

# Paths
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR        = os.path.join(ROOT, "data", "kb")
INDEX_DIR     = os.path.join(ROOT, "data", "processed", "faiss_index")
ABUSE_MODEL   = os.path.join(ROOT, "models", "abuse_clf")  # saved after training

# Models
EMB_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# LLM (OpenAI-compatible)
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL     = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_API_KEY   = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

ABUSE_LABELS  = ["domestic_abuse", "workplace_harassment", "coercive_control", "non_abuse"]
RISK_LEVELS   = ["low", "medium", "high"]
```

`src/preprocess.py`:
```python
import re

URL = re.compile(r"http\S+|www\.\S+")
WS  = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Light cleaning that preserves meaning (don't strip negations/emotion)."""
    text = text.strip()
    text = URL.sub(" ", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)   # keep basic punctuation
    text = WS.sub(" ", text)
    return text.strip()

# Map the 6 emotions -> distress level used by Agent 4
EMOTION_TO_DISTRESS = {
    "fear": "high", "anger": "high", "sadness": "medium",
    "surprise": "medium", "disgust": "medium",
    "joy": "low", "love": "low", "neutral": "low",
}
```

---

## 5. Agent 1 — Situation Listener

Two heads: (a) **binary abuse detector** (trained on Jigsaw), (b) **distress level** (ready emotion model), (c) **fine-grained category** via few-shot LLM.

### 5a. Train the binary abuse detector — `train/train_abuse_clf.py`
```python
"""Fine-tune DistilBERT to detect abusive/harassing text (binary).
Uses Jigsaw toxic-comment labels (toxic|threat|insult|identity_hate -> abusive=1)."""
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL = "distilbert-base-uncased"
OUT   = "models/abuse_clf"

ds = load_dataset("google/jigsaw_toxicity_pred")  # has 'comment_text' + 6 label cols
def to_binary(ex):
    abusive = int(any([ex["toxic"], ex["threat"], ex["insult"], ex["identity_hate"]]))
    return {"text": ex["comment_text"], "label": abusive}
ds = ds.map(to_binary, remove_columns=ds["train"].column_names)

# (optional) balance + subsample for speed on coursework hardware
ds["train"] = ds["train"].shuffle(seed=42).select(range(20000))

tok = AutoTokenizer.from_pretrained(MODEL)
def enc(b): return tok(b["text"], truncation=True, max_length=192)
ds = ds.map(enc, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="binary")
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "precision": pr, "recall": rc, "f1": f1}

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=2, per_device_train_batch_size=16,
    per_device_eval_batch_size=32, learning_rate=2e-5, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="f1", logging_steps=100)

trainer = Trainer(model=model, args=args,
                  train_dataset=ds["train"], eval_dataset=ds["test"],
                  tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metrics)
trainer.train()
trainer.save_model(OUT); tok.save_pretrained(OUT)
print("Saved abuse classifier to", OUT)
```

### 5b. The Listener agent — `src/agents/listener.py`
```python
from transformers import pipeline
from src.preprocess import clean_text, EMOTION_TO_DISTRESS
from src import config
from src.llm import few_shot_category

# Distress via ready emotion model (no training needed)
_emotion = pipeline("text-classification", model=config.EMOTION_MODEL, top_k=None)

# Binary abuse detector (after you train it). Falls back gracefully if missing.
try:
    _abuse = pipeline("text-classification", model=config.ABUSE_MODEL)
except Exception:
    _abuse = None

def analyze(text: str) -> dict:
    t = clean_text(text)

    # (1) distress level
    scores = _emotion(t)[0]
    top = max(scores, key=lambda s: s["score"])
    distress = EMOTION_TO_DISTRESS.get(top["label"].lower(), "low")

    # (2) abusive vs not (binary, trained)
    if _abuse:
        a = _abuse(t)[0]
        is_abuse = a["label"].endswith("1") or a["label"].lower() == "label_1"
        abuse_conf = a["score"]
    else:
        is_abuse, abuse_conf = None, None

    # (3) fine-grained category (few-shot, in-context learning)
    category = few_shot_category(t)

    return {
        "clean_text": t,
        "emotion": top["label"], "emotion_score": round(top["score"], 3),
        "distress": distress,
        "is_abuse": is_abuse, "abuse_conf": abuse_conf,
        "category": category,
    }
```

---

## 6. Knowledge base (RAG source) — `data/kb/`

Write **20–40 short, trustworthy docs** (one topic each, ~150–300 words). Keep them general and source-attributed. Suggested files:

```
kb_domestic_safety_planning.md
kb_domestic_support_orgs.md
kb_workplace_harassment_reporting.md
kb_workplace_rights_overview.md
kb_coercive_control_signs.md
kb_emergency_when_to_call.md
kb_evidence_documentation.md
kb_emotional_support_basics.md
... (collect from public, reputable sources and cite them)
```

Example `kb/kb_workplace_harassment_reporting.md`:
```markdown
# Reporting Workplace Harassment (General Information)

Source: General public guidance — NOT legal advice.

If you experience harassment at work, consider these general steps:
1. Keep a private record of incidents: dates, times, what was said/done, witnesses.
2. Review your organization's harassment or grievance policy (often in the staff
   handbook or HR portal).
3. Report to a trusted manager, HR, or a designated reporting officer in writing
   where possible, so there is a record.
4. If your employer has an anti-retaliation policy, note it — retaliation for
   reporting is often prohibited.
5. Keep copies of any communications. Store them somewhere private.

This information is general and may not reflect the laws or policies in your
country or workplace. For your specific situation, contact a qualified advisor,
a relevant labor authority, or a support organization in your area.
```

> Pull real content from reputable public sources (national helplines, labour authorities, recognised NGOs). **Cite each source inside the doc and in your report's references.** Do not invent statistics.

---

## 7. Agent 2 — RAG Retriever

### 7a. Build the FAISS index — `src/rag/build_index.py`
```python
import os, glob, pickle
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from src import config

def chunk(text, size=600, overlap=100):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def build():
    model = SentenceTransformer(config.EMB_MODEL)
    docs, meta = [], []
    for path in glob.glob(os.path.join(config.KB_DIR, "*.md")):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for j, c in enumerate(chunk(content)):
            docs.append(c)
            meta.append({"source": os.path.basename(path), "chunk": j})

    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])     # cosine (vectors normalized)
    index.add(np.asarray(emb, dtype="float32"))

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(config.INDEX_DIR, "kb.index"))
    with open(os.path.join(config.INDEX_DIR, "store.pkl"), "wb") as f:
        pickle.dump({"docs": docs, "meta": meta}, f)
    print(f"Indexed {len(docs)} chunks from {config.KB_DIR}")

if __name__ == "__main__":
    build()
```

### 7b. Retriever agent — `src/agents/retriever.py`
```python
import os, pickle
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from src import config

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(config.EMB_MODEL)
        self.index = faiss.read_index(os.path.join(config.INDEX_DIR, "kb.index"))
        with open(os.path.join(config.INDEX_DIR, "store.pkl"), "rb") as f:
            store = pickle.load(f)
        self.docs, self.meta = store["docs"], store["meta"]

    def retrieve(self, query: str, k: int = 4):
        q = self.model.encode([query], normalize_embeddings=True)
        scores, idx = self.index.search(np.asarray(q, dtype="float32"), k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            out.append({"text": self.docs[i], "score": float(s), **self.meta[i]})
        return out
```

---

## 8. LLM wrapper, Agent 3 (Planner), Agent 4 (Risk)

### 8a. LLM wrapper — `src/llm.py`
```python
from openai import OpenAI
from src import config

_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

def chat(system: str, user: str, temperature: float = 0.2) -> str:
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content.strip()

# Few-shot / in-context category classification (Agent 1 helper)
_FEWSHOT = """You classify a person's situation into exactly ONE label:
domestic_abuse | workplace_harassment | coercive_control | non_abuse
Reply with the label only.

Text: "My partner controls all the money and won't let me see my friends."
Label: coercive_control
Text: "My manager keeps texting me at night and hints I'll be fired if I complain."
Label: workplace_harassment
Text: "He shoved me again last night and I'm scared to go home."
Label: domestic_abuse
Text: "I'm stressed about my exam results next week."
Label: non_abuse
"""

def few_shot_category(text: str) -> str:
    out = chat("You are a careful text classifier. Output only the label.",
               _FEWSHOT + f'Text: "{text}"\nLabel:', temperature=0.0)
    label = out.strip().split()[0].lower()
    return label if label in config.ABUSE_LABELS else "non_abuse"
```

### 8b. Risk Evaluator — `src/agents/risk.py`
```python
import re

# Phrases suggesting immediate danger (expand for your domain; cite your rationale).
HIGH_RISK = re.compile(
    r"\b(kill|hurt me|weapon|gun|knife|strangl|chok|can'?t breathe|"
    r"threaten(ed|ing)? to kill|suicid|end it|right now|tonight|he'?s here)\b", re.I)
MED_RISK  = re.compile(r"\b(threat|scared|afraid|unsafe|follow(ed|ing)?|stalk)\b", re.I)

def evaluate(text: str, distress: str, is_abuse) -> dict:
    score = {"low": 0, "medium": 1, "high": 2}[distress]
    if MED_RISK.search(text):  score = max(score, 1)
    if HIGH_RISK.search(text): score = 2
    if is_abuse is False:      score = min(score, 1)   # de-escalate non-abuse
    level = ["low", "medium", "high"][score]
    return {
        "risk": level,
        "urgent_note": level == "high",
        "reason": "distress=%s; danger-cues=%s" % (
            distress, bool(HIGH_RISK.search(text)))
    }
```

### 8c. Safety Planner — `src/agents/planner.py`
```python
from src.llm import chat

SYSTEM = """You are SilenceBreaker, a supportive information assistant for people
who may be experiencing abuse or harassment. You are NOT a lawyer, doctor, or
emergency service. Rules:
- Be calm, warm, non-judgmental, and brief.
- Ground EVERY factual claim ONLY in the provided EVIDENCE. If evidence is weak
  or missing, say so and suggest contacting a local support service.
- Never give definitive legal/medical judgments. Use cautious wording.
- Offer practical, optional next steps as a short list.
- If RISK is high, begin with a clear line urging the person to seek immediate
  local help / emergency services.
- Do not store or ask for identifying personal data."""

def plan(user_text, category, distress, risk, evidence: list) -> str:
    ev = "\n\n".join(f"[{i+1}] (source: {e['source']}) {e['text']}"
                     for i, e in enumerate(evidence)) or "NONE"
    prompt = f"""SITUATION (category={category}, distress={distress}, risk={risk}):
\"\"\"{user_text}\"\"\"

EVIDENCE (use only this; cite as [1], [2] ...):
{ev}

Write a response with:
1. A brief empathetic acknowledgement (1–2 sentences).
2. "What this looks like" — a short, grounded summary.
3. "Possible next steps" — 3–5 optional, evidence-backed steps with [citations].
4. "Support resources" — point to organisations/resources from the evidence.
5. If risk is high, put an urgent help line FIRST.
End with one sentence reminding this is general info, not professional advice."""
    return chat(SYSTEM, prompt, temperature=0.3)
```

---

## 9. Orchestration with LangGraph — `src/graph.py`

Using LangGraph makes the multi-agent flow *visible* to examiners (worth marks).

```python
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from src.agents.listener import analyze
from src.agents.retriever import Retriever
from src.agents.risk import evaluate
from src.agents.planner import plan

_retriever = Retriever()

class State(TypedDict):
    text: str
    analysis: Optional[dict]
    evidence: Optional[List[dict]]
    risk: Optional[dict]
    response: Optional[str]

def n_listen(s: State):
    return {"analysis": analyze(s["text"])}

def n_retrieve(s: State):
    a = s["analysis"]
    query = f"{a['category']} {a['clean_text']}"
    return {"evidence": _retriever.retrieve(query, k=4)}

def n_risk(s: State):
    a = s["analysis"]
    return {"risk": evaluate(a["clean_text"], a["distress"], a["is_abuse"])}

def n_plan(s: State):
    a, r = s["analysis"], s["risk"]
    return {"response": plan(s["text"], a["category"], a["distress"],
                             r["risk"], s["evidence"])}

def build_graph():
    g = StateGraph(State)
    g.add_node("listener", n_listen)
    g.add_node("retriever", n_retrieve)
    g.add_node("risk", n_risk)
    g.add_node("planner", n_plan)
    g.set_entry_point("listener")
    g.add_edge("listener", "retriever")
    g.add_edge("retriever", "risk")
    g.add_edge("risk", "planner")
    g.add_edge("planner", END)
    return g.compile()

APP = build_graph()

def run(text: str) -> dict:
    final = APP.invoke({"text": text})
    a, r = final["analysis"], final["risk"]
    return {
        "category": a["category"], "distress": a["distress"],
        "emotion": a["emotion"], "is_abuse": a["is_abuse"],
        "risk": r["risk"], "urgent": r["urgent_note"],
        "evidence": final["evidence"], "response": final["response"],
    }

if __name__ == "__main__":
    import json
    out = run("My manager keeps sending inappropriate messages and threatens "
              "my job if I report it.")
    print(json.dumps(out, indent=2))
```

---

## 10. Streamlit demo — `app/streamlit_app.py`

```python
import streamlit as st
from src.graph import run

st.set_page_config(page_title="SilenceBreaker", page_icon="🛡️", layout="wide")
st.title("🛡️ SilenceBreaker — Support Assistant (Prototype)")
st.caption("General information only. Not legal, medical, or emergency advice.")

txt = st.text_area("Describe what's happening (you can stay anonymous):",
                   height=140,
                   placeholder="e.g. My manager keeps messaging me late at night...")

if st.button("Get support", type="primary") and txt.strip():
    with st.spinner("Analysing..."):
        r = run(txt)

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
            st.markdown(f"**[{i}] {e['source']}** · score={e['score']:.2f}")
            st.write(e["text"])

    st.divider()
    st.caption("This prototype does not store your data. It is not a substitute "
               "for professional or emergency help.")
```

Run it:
```bash
python -m src.rag.build_index        # build the index first
streamlit run app/streamlit_app.py
```

---

## 11. Evaluation (15 marks — do all four)

### 11a. Classifier metrics — `evaluation/eval_classifier.py`
```python
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

clf = pipeline("text-classification", model="models/abuse_clf")
ds = load_dataset("google/jigsaw_toxicity_pred")["test"].shuffle(seed=0).select(range(2000))

def gold(ex):
    return int(any([ex["toxic"], ex["threat"], ex["insult"], ex["identity_hate"]]))

y_true, y_pred = [], []
for ex in ds:
    y_true.append(gold(ex))
    p = clf(ex["comment_text"][:512])[0]["label"]
    y_pred.append(1 if p.lower().endswith("1") else 0)

print(classification_report(y_true, y_pred, digits=3))
print(confusion_matrix(y_true, y_pred))
```

### 11b. Retrieval P@k / hit-rate — `evaluation/eval_retrieval.py`
```python
"""Build ~15 test queries with the KB filename you EXPECT to be relevant.
Measures hit@k and precision@k."""
from src.agents.retriever import Retriever
r = Retriever()

GOLD = [
    ("how do I report harassment at work", "kb_workplace_harassment_reporting.md"),
    ("my partner controls my money and friends", "kb_coercive_control_signs.md"),
    ("I think I'm in danger tonight", "kb_emergency_when_to_call.md"),
    # ... add ~15 total
]

def hit_at_k(k=4):
    hits, p_sum = 0, 0
    for q, gold in GOLD:
        srcs = [d["source"] for d in r.retrieve(q, k)]
        hit = gold in srcs
        hits += int(hit)
        p_sum += srcs.count(gold) / k
    print(f"hit@{k} = {hits/len(GOLD):.3f}   precision@{k} = {p_sum/len(GOLD):.3f}")

hit_at_k(4)
```

### 11c. Faithfulness (% of claims grounded) — `evaluation/eval_faithfulness.py`
```python
"""LLM-as-judge: does each sentence in the response follow from the evidence?
Report faithfulness = supported_sentences / total_sentences."""
from src.llm import chat
from src.graph import run
import re

JUDGE = """You are a strict fact-checker. Given EVIDENCE and a SENTENCE, answer
SUPPORTED if the sentence's factual claims are entailed by the evidence (or it is
a generic empathetic/safety statement), else UNSUPPORTED. One word only."""

def faithfulness(text):
    out = run(text)
    ev = "\n".join(e["text"] for e in out["evidence"])
    sents = [s for s in re.split(r"(?<=[.!?])\s+", out["response"]) if len(s) > 15]
    ok = 0
    for s in sents:
        v = chat(JUDGE, f"EVIDENCE:\n{ev}\n\nSENTENCE: {s}", temperature=0.0)
        ok += int("SUPPORTED" in v.upper() and "UNSUPPORTED" not in v.upper())
    print(f"faithfulness = {ok}/{len(sents)} = {ok/max(len(sents),1):.2f}")

faithfulness("My manager threatens my job if I report his messages.")
```

### 11d. **Ablation study (the big mark-booster)** — `evaluation/ablation.py`
```python
"""Compare three systems on the SAME test prompts:
  A) Baseline LLM only (no RAG, no agents)
  B) RAG chatbot (retrieval + single LLM call)
  C) Full multi-agent SilenceBreaker (this repo)
Score each on: faithfulness, risk-flag accuracy, retrieval use, human rubric (1-5)."""
from src.llm import chat
from src.graph import run
from src.agents.retriever import Retriever
r = Retriever()

TEST = [
 ("My partner won't let me leave the house or see anyone.", "coercive_control", "high"),
 ("HR ignored my complaint and now I get fewer shifts.",    "workplace_harassment", "medium"),
 ("I'm just stressed about moving apartments.",             "non_abuse", "low"),
 # add ~15 prompts with expected category + expected risk
]

def baseline(t):           # A
    return chat("You are a helpful assistant.", t)

def rag_chatbot(t):        # B
    ev = "\n".join(d["text"] for d in r.retrieve(t, 4))
    return chat("Answer using only this evidence; cite it.", f"{t}\n\nEVIDENCE:\n{ev}")

def full(t):               # C
    return run(t)["response"]

# Run all three, save outputs to CSV, then score with eval_faithfulness +
# a human rubric (clarity, safety, groundedness, helpfulness; 1–5 each).
for t, _, _ in TEST:
    print("PROMPT:", t)
    print(" A:", baseline(t)[:120], "...")
    print(" B:", rag_chatbot(t)[:120], "...")
    print(" C:", full(t)[:120], "...\n")
```

Put the results in a table in your report — e.g.:

| System | Faithfulness ↑ | Risk-flag Acc ↑ | Human rubric (1–5) ↑ |
|---|---|---|---|
| A) LLM only | 0.55 | 0.40 | 2.8 |
| B) RAG chatbot | 0.82 | 0.55 | 3.6 |
| C) Multi-agent (ours) | **0.93** | **0.85** | **4.4** |

*(Fill with your real numbers.)* This single table is what makes it look like a serious Advanced-AI project rather than "just a chatbot."

---

## 12. Report structure (map 1:1 to assignment) + team plan

**Report sections** (≤20 pages, Times New Roman 12pt, 1.5 spacing, APA refs):
1. Executive Summary (1 page)
2. Problem Statement & Motivation — abuse victims lack safe, understandable, organised guidance; you *organise* support info, you don't diagnose.
3. Literature Review — RAG, multi-agent LLM systems, toxicity/emotion classification, safety in NLP. (Cite Jigsaw, CARER emotion paper, RAG paper, etc.)
4. Methodology — the 4 agents + the 5 AI techniques (put the technique→module table here).
5. Implementation Details — architecture diagram, agent workflow figure, prompt-design table, tools.
6. Experimental Setup & Results — datasets, splits, hyperparameters.
7. Evaluation & Metrics — §11 results + the ablation table.
8. Discussion — limitations, challenges, lessons learned.
9. Conclusion & Future Work.
10. References (APA 7).
11. Appendices — code snippets, extra charts, **contribution table**.

**Technique → module table (paste into Methodology):**

| Module | AI technique(s) |
|---|---|
| Situation Listener | NLP preprocessing + Transformer classification (DistilBERT/RoBERTa) |
| Retriever | Sentence embeddings + vector search (RAG) |
| Category head | Few-shot / in-context learning (LLM) |
| Safety Planner | LLM + systematic prompt engineering |
| Risk Evaluator | Rule-based reasoning + classifier signal |

**4-member roles (4-week version of the PDF plan):**

| Member | Owns | Files |
|---|---|---|
| M1 — Data & docs lead | preprocessing, KB curation, report | `preprocess.py`, `data/kb/*`, report |
| M2 — Classification & risk | abuse classifier, emotion/distress, risk agent | `train_abuse_clf.py`, `listener.py`, `risk.py` |
| M3 — RAG | chunking, embeddings, FAISS, retrieval eval | `build_index.py`, `retriever.py`, `eval_retrieval.py` |
| M4 — Orchestration & UI | LangGraph, prompts, Streamlit, integration | `llm.py`, `planner.py`, `graph.py`, `streamlit_app.py` |

**Week plan:** W1 setup + baselines (classifier, KB, index, UI skeleton) → W2 full 4-agent pipeline → W3 evaluation + ablation → W4 polish, report, 5-min video.

---

## 13. Ethics & safety section (paste into Discussion)

**Limitations:** prototype only; not a replacement for legal/medical/emergency services; English-only; quality bounded by the knowledge base; classifiers can err.

**Safety measures:** high-risk inputs trigger an urgent-help banner *first*; generation is grounded only in retrieved evidence; no definitive legal judgments; no storage of personal data in the demo; cautious, non-judgmental tone enforced by the system prompt.

**Defensible novelty statement:** "A multi-agent, risk-aware, evidence-grounded support assistant that combines emotion-aware situation understanding, retrieval of verified support knowledge, and structured, safety-oriented response generation." (Don't claim "no one has ever done this.")

---

## 14. Quick-start checklist

```bash
# 1. install
pip install -r requirements.txt
# 2. add 20–40 docs to data/kb/  (cite sources!)
# 3. build retrieval index
python -m src.rag.build_index
# 4. (optional but recommended) train binary abuse detector
python train/train_abuse_clf.py
# 5. set .env with your free Groq/OpenAI key
# 6. test the pipeline
python -m src.graph
# 7. run the demo
streamlit run app/streamlit_app.py
# 8. run evaluations + ablation, paste tables into the report
python evaluation/eval_classifier.py
python evaluation/eval_retrieval.py
python evaluation/eval_faithfulness.py
python evaluation/ablation.py
```

That's the whole project. Build the binary classifier + RAG + few-shot category + risk + ablation, write the report to the exact structure above, and you have a submission targeting the top band across every rubric criterion.
