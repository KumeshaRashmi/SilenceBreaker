"""Evaluate the trained binary abuse classifier on a Jigsaw test subset.

Run:
    python -m evaluation.eval_classifier
"""
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix


def main(n=2000):
    clf = pipeline("text-classification", model="models/abuse_clf")
    ds = load_dataset("cardiffnlp/tweet_eval", "hate")["test"]
    ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))

    def gold(ex):
        return int(ex["label"] == 1)

    y_true, y_pred = [], []
    for ex in ds:
        y_true.append(gold(ex))
        label = clf(ex["text"][:512])[0]["label"]
        y_pred.append(1 if label.lower().endswith("1") else 0)

    print(classification_report(y_true, y_pred, digits=3,
                                target_names=["non_abuse", "abuse"]))
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
