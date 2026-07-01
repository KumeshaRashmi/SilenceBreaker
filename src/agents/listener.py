"""Agent 1 - Situation Listener.

Analyses the user's text and returns:
  - emotion + distress level (ready emotion model, or keyword fallback)
  - is_abuse / confidence  (trained DistilBERT binary head, if available)
  - fine-grained category  (few-shot LLM, or rule-based fallback)
"""
from src import config
from src.preprocess import clean_text, EMOTION_TO_DISTRESS
from src.llm import few_shot_category

# ---------------------------------------------------------------------------
# Lazy model loading so importing this module is cheap and never crashes.
# ---------------------------------------------------------------------------
_emotion = None
_abuse = None
_loaded = False


def _load():
    global _emotion, _abuse, _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from transformers import pipeline
        _emotion = pipeline("text-classification",
                            model=config.EMOTION_MODEL, top_k=None)
    except Exception as exc:
        print(f"[listener] emotion model unavailable ({exc}); keyword fallback.")
        _emotion = None
    try:
        from transformers import pipeline
        _abuse = pipeline("text-classification", model=config.ABUSE_MODEL)
    except Exception:
        _abuse = None   # classifier not trained yet -> skip binary head


_DISTRESS_WORDS = {
    "high":   ["scared", "terrified", "afraid", "danger", "kill", "hurt",
               "threat", "panic", "help me"],
    "medium": ["anxious", "worried", "stressed", "upset", "unsafe", "angry"],
}


def _keyword_distress(text: str):
    low = text.lower()
    for word in _DISTRESS_WORDS["high"]:
        if word in low:
            return "neutral", "high"
    for word in _DISTRESS_WORDS["medium"]:
        if word in low:
            return "neutral", "medium"
    return "neutral", "low"


def analyze(text: str) -> dict:
    _load()
    t = clean_text(text)

    # (1) distress / emotion
    if _emotion is not None:
        scores = _emotion(t)[0]
        top = max(scores, key=lambda s: s["score"])
        emotion, emo_score = top["label"], round(float(top["score"]), 3)
        distress = EMOTION_TO_DISTRESS.get(emotion.lower(), "low")
    else:
        emotion, distress = _keyword_distress(t)
        emo_score = None

    # (2) abusive vs not (trained binary head, optional)
    if _abuse is not None:
        results = _abuse(t, top_k=None)
        abuse_score = next(r["score"] for r in results if r["label"].lower().endswith("1"))
        abuse_conf = round(float(abuse_score), 3)
        is_abuse = abuse_score >= config.ABUSE_THRESHOLD
    else:
        is_abuse, abuse_conf = None, None

    # (3) fine-grained category (few-shot LLM / rule fallback)
    category = few_shot_category(t)

    return {
        "clean_text": t,
        "emotion": emotion,
        "emotion_score": emo_score,
        "distress": distress,
        "is_abuse": is_abuse,
        "abuse_conf": abuse_conf,
        "category": category,
    }
