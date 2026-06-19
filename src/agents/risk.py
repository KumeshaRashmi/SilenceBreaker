"""Agent 4 - Risk Evaluator.

Combines the distress level (Agent 1) with rule-based danger cues to produce a
low / medium / high escalation label. High risk triggers an urgent-help note in
the UI and pushes that note to the top of the generated response.
"""
import re

# Phrases suggesting immediate danger. Expand and justify these in your report.
HIGH_RISK = re.compile(
    r"\b(kill|kills|killed|hurt me|weapon|gun|knife|strangl\w*|chok\w*|"
    r"can'?t breathe|threaten(?:ed|ing)? to kill|suicid\w*|end it all|"
    r"right now|tonight|he'?s here|she'?s here|going to hurt)\b", re.I)

MED_RISK = re.compile(
    r"\b(threat\w*|scared|afraid|unsafe|follow(?:ed|ing)?|stalk\w*|"
    r"trapped|can'?t leave|isolat\w*)\b", re.I)


def evaluate(text: str, distress: str, is_abuse) -> dict:
    base = {"low": 0, "medium": 1, "high": 2}.get(distress, 0)
    score = base
    if MED_RISK.search(text):
        score = max(score, 1)
    if HIGH_RISK.search(text):
        score = 2
    # Note: abuse classifier trained on hate-speech data; do NOT let it
    # override explicit danger cues — only use it as a soft downgrade on low scores.

    level = ["low", "medium", "high"][score]
    return {
        "risk": level,
        "urgent_note": level == "high",
        "reason": f"distress={distress}; danger_cues={bool(HIGH_RISK.search(text))}; "
                  f"warning_cues={bool(MED_RISK.search(text))}",
    }
