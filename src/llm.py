"""LLM wrapper (OpenAI-compatible) plus offline fallbacks.

Two public helpers are used by the agents:
  - few_shot_category(text): in-context-learning category classifier (Agent 1)
  - chat(system, user):      generic chat call (Agent 3 planner, judges)

If no API key is set (config.OFFLINE), both degrade to deterministic, rule-based
behaviour so the full pipeline still runs and is reproducible for grading.
"""
import re
import time
from src import config

# ---------------------------------------------------------------------------
# Online client (lazy import so the repo runs without `openai` in offline mode)
# ---------------------------------------------------------------------------
_client = None
if not config.OFFLINE:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    except Exception as exc:  # pragma: no cover
        print(f"[llm] Could not init OpenAI client ({exc}); using OFFLINE mode.")
        config.OFFLINE = True


# Free-tier Groq is capped at 6000 tokens/minute. A short fixed pause between
# calls keeps evaluation runs under that cap instead of relying purely on
# reactive backoff after a 429 (which still wastes the failed attempt).
_MIN_SECONDS_BETWEEN_CALLS = 2.5
_last_call_ts = 0.0


def _pace():
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_ts = time.time()


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    """Single chat completion with retry on rate limit. Falls back offline."""
    if config.OFFLINE or _client is None:
        return _offline_chat(system, user)
    for attempt in range(6):
        _pace()
        try:
            resp = _client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                wait = min(2 ** attempt, 60)
                print(f"[llm] Rate limit hit, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return _offline_chat(system, user)


# ---------------------------------------------------------------------------
# Few-shot category classification (in-context learning)
# ---------------------------------------------------------------------------
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
    if config.OFFLINE or _client is None:
        return _rule_category(text)
    out = chat(
        "You are a careful text classifier. Output only the label.",
        _FEWSHOT + f'Text: "{text}"\nLabel:',
        temperature=0.0,
    )
    label = out.strip().split()[0].lower() if out.strip() else "non_abuse"
    return label if label in config.ABUSE_LABELS else "non_abuse"


# ---------------------------------------------------------------------------
# Offline fallbacks (deterministic; documented as a degraded mode in report)
# ---------------------------------------------------------------------------
_WORKPLACE = re.compile(r"\b(manager|boss|hr|colleague|co-?worker|supervisor|"
                        r"office|job|fired|workplace|shift|promotion)\b", re.I)
_DOMESTIC  = re.compile(r"\b(partner|husband|wife|boyfriend|girlfriend|spouse|"
                        r"hit|shov\w*|punch\w*|slap\w*|kick\w*|"
                        r"hurt me|go home|at home)\b", re.I)
_COERCIVE  = re.compile(r"\b(control|won'?t let me|isolate|monitor|track|"
                        r"my money|my phone|permission|allowed to)\b", re.I)
_ABUSE_HINT = re.compile(r"\b(abuse|harass|threat|scared|afraid|unsafe|"
                         r"hurt|hit|stalk|control)\b", re.I)


def _rule_category(text: str) -> str:
    if _COERCIVE.search(text):
        return "coercive_control"
    if _WORKPLACE.search(text) and _ABUSE_HINT.search(text):
        return "workplace_harassment"
    if _DOMESTIC.search(text):
        return "domestic_abuse"
    if _WORKPLACE.search(text):
        return "workplace_harassment"
    return "non_abuse"


def _offline_chat(system: str, user: str) -> str:
    """Template planner used when no LLM is configured.

    It extracts the EVIDENCE block from the planner prompt and stitches a calm,
    grounded, clearly-templated response so the demo works without an API key.
    """
    evidence = _extract_evidence(user)
    risk = "high" if "risk=high" in user.lower() else (
        "medium" if "risk=medium" in user.lower() else "low")

    lines = []
    if risk == "high":
        lines.append("If you are in immediate danger, please contact your local "
                     "emergency number or a crisis helpline right now.\n")
    lines.append("I'm really glad you reached out. What you're describing sounds "
                 "difficult, and you deserve support.\n")
    lines.append("**What this looks like**")
    lines.append("Based on the information available, here is some general "
                 "guidance drawn from trusted support material.\n")
    lines.append("**Possible next steps**")
    if evidence:
        for i, ev in enumerate(evidence[:4], 1):
            snippet = ev.strip().replace("\n", " ")
            lines.append(f"- {snippet[:220]}... [{i}]")
    else:
        lines.append("- Consider contacting a local support organisation that "
                     "can give advice specific to your situation.")
    lines.append("\n**Support resources**")
    lines.append("Reach out to a recognised support organisation or helpline in "
                 "your area; they can offer confidential, situation-specific help.")
    lines.append("\nThis is general information, not professional or legal advice.")
    return "\n".join(lines)


def _extract_evidence(prompt: str):
    """Pull the numbered evidence chunks out of a planner prompt."""
    m = re.search(r"EVIDENCE.*?:\n(.*?)(?:\n\nWrite|\Z)", prompt, re.S)
    if not m:
        return []
    block = m.group(1)
    if block.strip() == "NONE":
        return []
    # split on the "[n] (source: ...)" markers
    parts = re.split(r"\[\d+\]\s*\(source:[^)]*\)\s*", block)
    return [p for p in parts if p.strip()]