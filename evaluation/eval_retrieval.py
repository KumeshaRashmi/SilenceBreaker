"""Evaluate retrieval quality: hit@k and precision@k against a gold set.

Run (after building the index):
    python -m evaluation.eval_retrieval
"""
from src.agents.retriever import Retriever
from evaluation.test_prompts import RETRIEVAL_GOLD


def evaluate(k=4):
    r = Retriever()
    hits, p_sum = 0, 0.0
    for query, gold in RETRIEVAL_GOLD:
        srcs = [d["source"] for d in r.retrieve(query, k)]
        hits += int(gold in srcs)
        p_sum += srcs.count(gold) / k
    n = len(RETRIEVAL_GOLD)
    print(f"Queries: {n}")
    print(f"hit@{k}        = {hits / n:.3f}")
    print(f"precision@{k}  = {p_sum / n:.3f}")


if __name__ == "__main__":
    evaluate(4)
