"""Agent 2 - RAG Retriever.

Loads the FAISS index built by src/rag/build_index.py and returns the top-k most
relevant knowledge-base chunks for a query (cosine similarity over MiniLM
embeddings).
"""
import os
import pickle
import numpy as np
from src import config


class Retriever:
    def __init__(self):
        import faiss
        from src.embedder import MiniLMEmbedder
        index_path = os.path.join(config.INDEX_DIR, "kb.index")
        store_path = os.path.join(config.INDEX_DIR, "store.pkl")
        if not (os.path.exists(index_path) and os.path.exists(store_path)):
            raise FileNotFoundError(
                "FAISS index not found. Run:  python -m src.rag.build_index")
        self.model = MiniLMEmbedder(config.EMB_MODEL)
        self.index = faiss.read_index(index_path)
        with open(store_path, "rb") as f:
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
