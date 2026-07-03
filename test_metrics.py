#!/usr/bin/env python3
from src.agents.retriever import Retriever
from evaluation.test_prompts import RETRIEVAL_GOLD

for k in [3, 4]:
    r = Retriever()
    hits, p_sum = 0, 0.0
    correct_queries = []
    for i, (query, gold) in enumerate(RETRIEVAL_GOLD):
        srcs = [d["source"] for d in r.retrieve(query, k)]
        is_hit = int(gold in srcs)
        hits += is_hit
        p_sum += srcs.count(gold) / k
        if is_hit:
            correct_queries.append(i+1)
    n = len(RETRIEVAL_GOLD)
    print(f"\nk={k}:")
    print(f"  hit@{k}       = {hits / n:.3f}")
    print(f"  precision@{k} = {p_sum / n:.3f}")
    print(f"  Correct queries: {correct_queries}/{n}")
