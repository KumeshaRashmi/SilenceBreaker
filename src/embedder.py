"""Lightweight MiniLM embedder using `transformers` directly.

Replaces `sentence_transformers` to avoid the datasets->pandas->pyarrow
DLL crash on Windows. Produces identical embeddings (mean-pool + L2-norm).
"""
import numpy as np


class MiniLMEmbedder:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, batch_size: int = 32):
        if isinstance(texts, str):
            texts = [texts]
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        out = []
        for batch in batches:
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    padding=True, truncation=True, max_length=512)
            with self._torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            if normalize_embeddings:
                emb = self._torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.numpy())
        return np.vstack(out)
