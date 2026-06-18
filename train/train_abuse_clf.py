"""Fine-tune DistilBERT to detect abusive / harassing text (binary).

Uses the Jigsaw toxic-comment labels: a comment is 'abusive' (1) if any of
toxic / threat / insult / identity_hate is set, else 0.

Run:
    python train/train_abuse_clf.py
Output: models/abuse_clf/  (loaded automatically by Agent 1)
"""
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL = "distilbert-base-uncased"
OUT = "models/abuse_clf"
TRAIN_SUBSET = 20000          # reduce for faster training on coursework hardware


def main():
    ds = load_dataset("google/jigsaw_toxicity_pred")

    def to_binary(ex):
        abusive = int(any([ex["toxic"], ex["threat"],
                           ex["insult"], ex["identity_hate"]]))
        return {"text": ex["comment_text"], "label": abusive}

    ds = ds.map(to_binary, remove_columns=ds["train"].column_names)
    ds["train"] = ds["train"].shuffle(seed=42).select(range(TRAIN_SUBSET))

    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(b):
        return tok(b["text"], truncation=True, max_length=192)

    ds = ds.map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        pr, rc, f1, _ = precision_recall_fscore_support(
            p.label_ids, preds, average="binary", zero_division=0)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "precision": pr, "recall": rc, "f1": f1}

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=2,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=100)

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics)

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("Saved abuse classifier to", OUT)


if __name__ == "__main__":
    main()
