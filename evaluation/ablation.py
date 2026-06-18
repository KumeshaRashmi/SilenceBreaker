"""Ablation study (the biggest mark-booster):
  A) Baseline LLM only       - no retrieval, no agents
  B) RAG chatbot             - retrieval + single LLM call
  C) Full multi-agent system - this repo

Saves per-prompt outputs to evaluation/ablation_outputs.csv and prints a summary
table (category accuracy + risk-flag accuracy for the full system; outputs for
A/B for qualitative + faithfulness scoring).

Run:
    python -m evaluation.ablation
"""
import csv
import os
from src.llm import chat
from src.graph import run
from src.agents.retriever import Retriever
from evaluation.test_prompts import SCENARIOS

_r = None


def _ret():
    global _r
    if _r is None:
        _r = Retriever()
    return _r


def baseline_llm(text):                       # A
    return chat("You are a helpful assistant.", text)


def rag_chatbot(text):                        # B
    ev = "\n".join(d["text"] for d in _ret().retrieve(text, 4))
    return chat("Answer using ONLY this evidence and cite it as [1],[2].",
                f"{text}\n\nEVIDENCE:\n{ev}")


def full_system(text):                        # C
    return run(text)


def main():
    out_path = os.path.join(os.path.dirname(__file__), "ablation_outputs.csv")
    cat_ok = risk_ok = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "exp_category", "exp_risk",
                    "A_baseline", "B_rag", "C_category", "C_risk", "C_response"])
        for text, exp_cat, exp_risk in SCENARIOS:
            a = baseline_llm(text)
            b = rag_chatbot(text)
            c = full_system(text)
            cat_ok += int(c["category"] == exp_cat)
            risk_ok += int(c["risk"] == exp_risk)
            w.writerow([text, exp_cat, exp_risk, a, b,
                        c["category"], c["risk"], c["response"]])

    n = len(SCENARIOS)
    print(f"Wrote {out_path}")
    print("\n=== Full system (C) automatic metrics ===")
    print(f"Category accuracy  = {cat_ok / n:.3f}")
    print(f"Risk-flag accuracy = {risk_ok / n:.3f}")
    print("\nNow score A vs B vs C for faithfulness (eval_faithfulness.py) and "
          "a human rubric (clarity, safety, groundedness, helpfulness; 1-5), "
          "and put the comparison table in your report.")


if __name__ == "__main__":
    main()
