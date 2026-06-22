"""Evaluate the trained binary abuse classifier on tweet_eval hate test split.

Uses the same dataset the model was trained on (cardiffnlp/tweet_eval "hate").
label 0 = non-abusive, label 1 = abusive/hate speech.

Prints metrics at the default threshold (0.5) AND at the F1-optimal threshold
found automatically via precision_recall_curve.

Run:
    python -m evaluation.eval_classifier
"""
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve


def main(n=2000):
    clf = pipeline("text-classification", model="models/abuse_clf", top_k=None)
    ds = load_dataset("cardiffnlp/tweet_eval", "hate")["test"]
    ds = ds.shuffle(seed=0).select(range(min(n, len(ds))))

    y_true, abuse_scores = [], []
    for ex in ds:
        y_true.append(int(ex["label"]))
        all_scores = clf(ex["text"][:512])[0]
        score = next(r["score"] for r in all_scores if r["label"].lower().endswith("1"))
        abuse_scores.append(score)

    # ── Default threshold ──────────────────────────────────────────────────────
    y_pred_default = [1 if s >= 0.5 else 0 for s in abuse_scores]
    print("=" * 55)
    print("DEFAULT THRESHOLD (0.5)")
    print("=" * 55)
    print(classification_report(y_true, y_pred_default, digits=3,
                                target_names=["non_abuse", "abuse"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred_default))

    # ── F1-optimal threshold ───────────────────────────────────────────────────
    precision, recall, thresholds = precision_recall_curve(y_true, abuse_scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    best_thresh = float(thresholds[best_idx])

    y_pred_opt = [1 if s >= best_thresh else 0 for s in abuse_scores]
    print(f"\n{'=' * 55}")
    print(f"OPTIMAL THRESHOLD ({best_thresh:.3f})  ← use this in production")
    print("=" * 55)
    print(classification_report(y_true, y_pred_opt, digits=3,
                                target_names=["non_abuse", "abuse"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred_opt))
    print(f"\n>>> Set ABUSE_THRESHOLD={best_thresh:.3f} in src/config.py")


if __name__ == "__main__":
    main()
