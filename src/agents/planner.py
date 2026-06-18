"""Agent 3 - Safety Planner.

Generates a calm, supportive, evidence-grounded response. The system prompt
enforces the safety rules (no legal/medical judgments, ground claims in the
retrieved evidence, urgent help first when risk is high). In offline mode the
LLM wrapper substitutes a template that reuses the same evidence.
"""
from src.llm import chat

SYSTEM = """You are SilenceBreaker, a supportive information assistant for people
who may be experiencing abuse or harassment. You are NOT a lawyer, doctor, or
emergency service. Rules:
- Be calm, warm, non-judgmental, and brief.
- Ground EVERY factual claim ONLY in the provided EVIDENCE. If evidence is weak
  or missing, say so and suggest contacting a local support service.
- Never give definitive legal or medical judgments. Use cautious wording.
- Offer practical, OPTIONAL next steps as a short list.
- If RISK is high, begin with a clear line urging the person to seek immediate
  local help / emergency services.
- Do not store or ask for identifying personal data."""


def plan(user_text: str, category: str, distress: str, risk: str,
         evidence: list) -> str:
    ev = "\n\n".join(
        f"[{i+1}] (source: {e['source']}) {e['text']}"
        for i, e in enumerate(evidence)
    ) or "NONE"

    prompt = f"""SITUATION (category={category}, distress={distress}, risk={risk}):
\"\"\"{user_text}\"\"\"

EVIDENCE (use only this; cite as [1], [2] ...):
{ev}

Write a response with:
1. A brief empathetic acknowledgement (1-2 sentences).
2. "What this looks like" - a short, grounded summary.
3. "Possible next steps" - 3-5 optional, evidence-backed steps with [citations].
4. "Support resources" - point to organisations/resources from the evidence.
5. If risk is high, put an urgent help line FIRST.
End with one sentence reminding this is general info, not professional advice."""

    return chat(SYSTEM, prompt, temperature=0.3)
