"""Text pre-processing utilities and the emotion -> distress mapping.

The cleaning is deliberately light: we keep punctuation, negations and emotional
words because they carry signal for both classification and risk scoring.
"""
import re

_URL = re.compile(r"http\S+|www\.\S+")
_WS  = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light, meaning-preserving cleaning."""
    if not text:
        return ""
    text = text.strip()
    text = _URL.sub(" ", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)   # keep basic punctuation
    text = _WS.sub(" ", text)
    return text.strip()


# Map the emotion-model labels to a 3-level distress scale used by Agent 4.
EMOTION_TO_DISTRESS = {
    "fear": "high", "anger": "high",
    "sadness": "medium", "surprise": "medium", "disgust": "medium",
    "joy": "low", "love": "low", "neutral": "low",
}
