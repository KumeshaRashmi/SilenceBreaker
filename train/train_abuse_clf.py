"""Fine-tune DistilBERT to detect abusive / harassing text (binary).

Uses cardiffnlp/tweet_eval "hate" split (standard Parquet format on HuggingFace).
label 0 = non-abusive, 1 = abusive/hate speech.

Run:
    python train/train_abuse_clf.py
Output: models/abuse_clf/  (loaded automatically by Agent 1)
"""
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

MODEL = "distilbert-base-uncased"
OUT = "models/abuse_clf"


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the loss to handle imbalance."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = torch.tensor(self.class_weights, dtype=torch.float,
                              device=logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    ds = load_dataset("cardiffnlp/tweet_eval", "hate")

    # Compute class weights from training labels
    train_labels = np.array(ds["train"]["label"])
    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    print(f"Class weights: {dict(zip(classes.tolist(), weights.tolist()))}")

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
                "precision": float(pr), "recall": float(rc), "f1": float(f1)}

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50)

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
        class_weights=weights.tolist())

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("Saved abuse classifier to", OUT)


if __name__ == "__main__":
    main()
