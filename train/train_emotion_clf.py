"""(Optional) Fine-tune DistilBERT for emotion classification on dair-ai/emotion.

The pipeline uses the ready-made j-hartmann emotion model by default, so this is
optional. Train your own if you want a 'we fine-tuned a transformer' story with
your own metrics in the report.

Run:
    python train/train_emotion_clf.py
Output: models/emotion_clf/
"""
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

MODEL = "distilbert-base-uncased"
OUT = "models/emotion_clf"


def main():
    ds = load_dataset("dair-ai/emotion")
    num_labels = ds["train"].features["label"].num_classes

    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(b):
        return tok(b["text"], truncation=True, max_length=128)

    ds = ds.map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=num_labels)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "f1_macro": f1_score(p.label_ids, preds, average="macro")}

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=3,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        logging_steps=100)

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics)

    trainer.train()
    print(trainer.evaluate(ds["test"]))
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("Saved emotion classifier to", OUT)


if __name__ == "__main__":
    main()
