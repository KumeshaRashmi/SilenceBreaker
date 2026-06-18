"""Evaluate the full system:
  - category accuracy and risk-flag accuracy (vs expected labels)
  - faithfulness: fraction of response sentences supported by the evidence
    (LLM-as-judge; in offline mode this is skipped with a note)

Run:
    python -m evaluation.eval_faithfulness
"""
import re
from src import config
from src.graph import run
from src.llm import chat
from evaluation.test_prompts import SCENARIOS

JUDGE = """You are a strict fact-checker. Given EVIDENCE and a SENTENCE, answer
SUPPORTED if the sentence's factual claims are entailed by the evidence, or if it
is a generic empathetic / safety / disclaimer statement. Otherwise answer
UNSUPPORTED. Reply with one word only."""


def sentence_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 15]


def faithfulness(response, evidence):
    if config.OFFLINE:
        return None
    ev = "\n".join(e["text"] for e in evidence)
    sents = sentence_split(response)
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        v = chat(JUDGE, f"EVIDENCE:\n{ev}\n\nSENTENCE: {s}", temperature=0.0).upper()
        ok += int("SUPPORTED" in v and "UNSUPPORTED" not in v)
    return ok / len(sents)


def main():
    cat_ok = risk_ok = 0
    faith_scores = []
    for text, exp_cat, exp_risk in SCENARIOS:
        out = run(text)
        cat_ok += int(out["category"] == exp_cat)
        risk_ok += int(out["risk"] == exp_risk)
        f = faithfulness(out["response"], out["evidence"])
        if f is not None:
            faith_scores.append(f)

    n = len(SCENARIOS)
    print(f"Category accuracy   = {cat_ok / n:.3f}  ({cat_ok}/{n})")
    print(f"Risk-flag accuracy  = {risk_ok / n:.3f}  ({risk_ok}/{n})")
    if faith_scores:
        print(f"Faithfulness (mean) = {sum(faith_scores) / len(faith_scores):.3f}")
    else:
        print("Faithfulness skipped (OFFLINE mode - set an LLM key to measure).")


if __name__ == "__main__":
    main()
