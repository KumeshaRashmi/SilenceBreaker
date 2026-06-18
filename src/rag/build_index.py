"""Build the FAISS index from the knowledge-base documents in data/kb/.

Run once before using the retriever or the app:
    python -m src.rag.build_index
"""
import os
import glob
import pickle
import numpy as np
from src import config


def chunk(text: str, size: int = 220, overlap: int = 40):
    """Word-based chunking with overlap (keeps short KB docs mostly intact)."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += max(1, size - overlap)
    return chunks


def build():
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMB_MODEL)
    docs, meta = [], []
    paths = sorted(glob.glob(os.path.join(config.KB_DIR, "*.md")))
    if not paths:
        raise FileNotFoundError(f"No .md docs found in {config.KB_DIR}")

    for path in paths:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for j, c in enumerate(chunk(content)):
            docs.append(c)
            meta.append({"source": os.path.basename(path), "chunk": j})

    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])      # inner product == cosine (normalized)
    index.add(np.asarray(emb, dtype="float32"))

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(config.INDEX_DIR, "kb.index"))
    with open(os.path.join(config.INDEX_DIR, "store.pkl"), "wb") as f:
        pickle.dump({"docs": docs, "meta": meta}, f)
    print(f"Indexed {len(docs)} chunks from {len(paths)} docs -> {config.INDEX_DIR}")


if __name__ == "__main__":
    build()
