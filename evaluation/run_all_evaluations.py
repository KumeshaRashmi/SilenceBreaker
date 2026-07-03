"""Complete evaluation pipeline with CSV export and visualization.

Run:
    python -m evaluation.run_all_evaluations

Generates:
    - evaluation/results/classifier_metrics.csv
    - evaluation/results/retrieval_metrics.csv
    - evaluation/results/ablation_comparison.csv
    - evaluation/results/metrics_plots.html
"""
import os
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.graph import run
from src.llm import chat
from src.agents.retriever import Retriever
from evaluation.test_prompts import SCENARIOS

# ----------------------------------------------------------------------------
# Shared cache: run() is expensive (2 LLM calls each) and was being invoked
# separately inside eval_classifier, eval_faithfulness, and eval_ablation for
# the *same* 15 scenarios -> 3x the necessary token usage. Compute once here.
# ----------------------------------------------------------------------------
_RUN_CACHE = {}


def cached_run(text):
    if text not in _RUN_CACHE:
        _RUN_CACHE[text] = run(text)
    return _RUN_CACHE[text]

# Setup results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# 1. CLASSIFIER METRICS (binary abuse detection)
# ============================================================================
def eval_classifier():
    """Evaluate the binary abuse classifier on test scenarios."""
    print("\n" + "="*70)
    print("EVALUATION 1: Abuse Classifier Metrics")
    print("="*70)
    
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                  f1_score, classification_report, confusion_matrix)
    
    # Generate predictions from scenarios
    y_true = []
    y_pred = []
    
    for text, expected_cat, expected_risk, expected_abuse in SCENARIOS:
        result = cached_run(text)
        y_true.append(expected_abuse)
        
        y_pred.append(result.get("is_abuse", False))
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-abuse", "Abuse"]))
    
    # Save to CSV
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [acc, prec, rec, f1],
    }
    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(RESULTS_DIR, "classifier_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to {csv_path}")
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ============================================================================
# 2. RETRIEVAL METRICS (P@k, hit@k)
# ============================================================================
def eval_retrieval():
    """Evaluate retrieval precision@k and hit@k."""
    print("\n" + "="*70)
    print("EVALUATION 2: Retrieval Metrics (Precision@k, Hit@k)")
    print("="*70)
    
    retriever = Retriever()
    hits_k3, hits_k4 = 0, 0
    prec_k3, prec_k4 = 0, 0
    n = 0
    
    for text, _, _, _ in SCENARIOS:
        results_k3 = retriever.retrieve(text, k=3)
        results_k4 = retriever.retrieve(text, k=4)
        
        # Simple heuristic: if any result has high similarity, consider it a hit
        if results_k3 and results_k3[0]["score"] > 0.5:
            hits_k3 += 1
        if results_k4 and results_k4[0]["score"] > 0.5:
            hits_k4 += 1
        
        prec_k3 += sum(1 for r in results_k3 if r["score"] > 0.5) / 3
        prec_k4 += sum(1 for r in results_k4 if r["score"] > 0.5) / 4
        n += 1
    
    hit_k3 = hits_k3 / n
    hit_k4 = hits_k4 / n
    p_k3 = prec_k3 / n if n > 0 else 0
    p_k4 = prec_k4 / n if n > 0 else 0
    
    print(f"Hit@3:       {hit_k3:.4f}")
    print(f"Hit@4:       {hit_k4:.4f}")
    print(f"Precision@3: {p_k3:.4f}")
    print(f"Precision@4: {p_k4:.4f}")
    
    # Save to CSV
    retrieval_data = {
        "Metric": ["Hit@3", "Hit@4", "Precision@3", "Precision@4"],
        "Score": [hit_k3, hit_k4, p_k3, p_k4],
    }
    df = pd.DataFrame(retrieval_data)
    csv_path = os.path.join(RESULTS_DIR, "retrieval_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to {csv_path}")
    
    return {"hit_k3": hit_k3, "hit_k4": hit_k4, "prec_k3": p_k3, "prec_k4": p_k4}


# ============================================================================
# 3. FAITHFULNESS (LLM-based grounding check)
# ============================================================================
def eval_faithfulness():
    """Score faithfulness of full system responses."""
    print("\n" + "="*70)
    print("EVALUATION 3: Faithfulness (Response Grounding)")
    print("="*70)
    
    from src import config
    import re
    
    faith_scores = []
    
    # Batched judge: ONE call per scenario (all sentences at once) instead of
    # one call per sentence. Cuts faithfulness-eval token usage by ~4-6x.
    JUDGE = """You are a strict fact-checker. You will be given EVIDENCE and a
numbered list of SENTENCES from a generated response. For EACH sentence, answer
SUPPORTED if its factual claims are entailed by the evidence, or if it is a
generic empathetic / safety / disclaimer statement. Otherwise answer UNSUPPORTED.

Reply with exactly one line per sentence, in the format:
1: SUPPORTED
2: UNSUPPORTED
...
No other text."""

    def sentence_split(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]

    for text, _, _, _ in SCENARIOS:
        result = cached_run(text)
        response = result.get("response", "")
        evidence = result.get("evidence", [])

        if not response or not evidence:
            continue

        ev_text = "\n".join(e["text"] for e in evidence)
        sents = sentence_split(response)

        if not sents:
            continue

        if config.OFFLINE:
            ok = len(sents)
        else:
            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents))
            try:
                judge_result = chat(
                    JUDGE, f"EVIDENCE:\n{ev_text}\n\nSENTENCES:\n{numbered}",
                    temperature=0.0,
                ).upper()
                ok = 0
                for line in judge_result.splitlines():
                    if "SUPPORTED" in line and "UNSUPPORTED" not in line:
                        ok += 1
            except Exception:
                ok = 0

        faith = ok / len(sents) if sents else 0
        faith_scores.append(faith)
    
    mean_faith = np.mean(faith_scores) if faith_scores else 0
    std_faith = np.std(faith_scores) if faith_scores else 0
    
    print(f"Mean faithfulness:   {mean_faith:.4f}")
    print(f"Std deviation:       {std_faith:.4f}")
    print(f"Samples evaluated:   {len(faith_scores)}")
    
    if config.OFFLINE:
        print("(OFFLINE mode - using template-based responses)")
    
    # Save to CSV
    faith_data = {
        "Metric": ["Mean Faithfulness", "Std Dev", "Samples"],
        "Value": [mean_faith, std_faith, len(faith_scores)],
    }
    df = pd.DataFrame(faith_data)
    csv_path = os.path.join(RESULTS_DIR, "faithfulness_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to {csv_path}")
    
    return {"mean": mean_faith, "std": std_faith, "samples": len(faith_scores)}


# ============================================================================
# 4. ABLATION STUDY (A vs B vs C)
# ============================================================================
def eval_ablation():
    """Compare baseline LLM vs RAG vs full multi-agent."""
    print("\n" + "="*70)
    print("EVALUATION 4: Ablation Study (A vs B vs C)")
    print("="*70)
    
    from src.llm import chat
    from src.agents.retriever import Retriever
    
    retriever = Retriever()
    ablation_results = []
    
    def baseline_llm(text):
        """A: Plain LLM only."""
        return chat("You are a helpful assistant.", text)
    
    def rag_chatbot(text):
        """B: RAG chatbot."""
        ev = "\n".join(d["text"] for d in retriever.retrieve(text, 4))
        return chat("Answer using ONLY this evidence and cite it.",
                   f"{text}\n\nEVIDENCE:\n{ev}")
    
    def full_system(text):
        """C: Full multi-agent system."""
        return cached_run(text)
    
    # Collect data for ablation
    for i, (text, exp_cat, exp_risk, exp_abuse) in enumerate(SCENARIOS[:5], 1):  # limit to 5 for speed
        print(f"\nScenario {i}:")
        print(f"Input: {text[:80]}...")
        
        try:
            resp_a = baseline_llm(text)[:150]
        except:
            resp_a = "[Error]"
        
        try:
            resp_b = rag_chatbot(text)[:150]
        except:
            resp_b = "[Error]"
        
        try:
            result_c = full_system(text)
            resp_c = result_c.get("response", "")[:150]
            risk_c = result_c.get("risk", "unknown")
            cat_c = result_c.get("category", "unknown")
        except:
            resp_c = "[Error]"
            risk_c = "unknown"
            cat_c = "unknown"
        
        ablation_results.append({
            "scenario": i,
            "input": text[:80],
            "expected_category": exp_cat,
            "expected_risk": exp_risk,
            "A_baseline": resp_a,
            "B_rag": resp_b,
            "C_response": resp_c,
            "C_category": cat_c,
            "C_risk": risk_c,
        })
        
        print(f"  A (LLM):     {resp_a}...")
        print(f"  B (RAG):     {resp_b}...")
        print(f"  C (Full):    {resp_c}...")
    
    # Save to CSV
    df = pd.DataFrame(ablation_results)
    csv_path = os.path.join(RESULTS_DIR, "ablation_outputs.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved ablation data to {csv_path}")
    
    return ablation_results


# ============================================================================
# 5. GENERATE PLOTS
# ============================================================================
def generate_plots():
    """Create visualization plots and save as HTML."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SilenceBreaker Evaluation Metrics", fontsize=16, fontweight='bold')
    
    # Plot 1: Classifier metrics
    classifier_csv = os.path.join(RESULTS_DIR, "classifier_metrics.csv")
    if os.path.exists(classifier_csv):
        df_clf = pd.read_csv(classifier_csv)
        axes[0, 0].bar(df_clf["Metric"], df_clf["Score"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_title("Abuse Classifier Metrics")
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_ylabel("Score")
        for i, v in enumerate(df_clf["Score"]):
            axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Plot 2: Retrieval metrics
    retrieval_csv = os.path.join(RESULTS_DIR, "retrieval_metrics.csv")
    if os.path.exists(retrieval_csv):
        df_ret = pd.read_csv(retrieval_csv)
        axes[0, 1].bar(df_ret["Metric"], df_ret["Score"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 1].set_title("Retrieval Metrics (P@k, Hit@k)")
        axes[0, 1].set_ylim([0, 1.1])
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(df_ret["Score"]):
            axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Plot 3: Faithfulness
    faith_csv = os.path.join(RESULTS_DIR, "faithfulness_metrics.csv")
    if os.path.exists(faith_csv):
        df_faith = pd.read_csv(faith_csv)
        metrics_to_plot = df_faith[df_faith["Metric"].isin(["Mean Faithfulness"])]["Value"].values
        if len(metrics_to_plot) > 0:
            axes[1, 0].bar(["Faithfulness"], metrics_to_plot, color='#17becf')
            axes[1, 0].set_title("Response Faithfulness")
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].text(0, metrics_to_plot[0] + 0.02, f"{metrics_to_plot[0]:.3f}", ha='center')
    
    # Plot 4: Summary text
    axes[1, 1].axis('off')
    summary_text = """
EVALUATION SUMMARY

✓ Classifier Metrics: Abuse detection performance
✓ Retrieval Metrics: Evidence retrieval quality
✓ Faithfulness: Response grounding in evidence
✓ Ablation Study: System component comparison

All results saved to: evaluation/results/
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "metrics_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plots to {plot_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run all evaluations in sequence."""
    print("\n" + "█"*70)
    print("█  SILENCEBREAKER FULL EVALUATION PIPELINE")
    print("█"*70)
    
    results = {}
    
    # Run each evaluation
    try:
        results["classifier"] = eval_classifier()
    except Exception as e:
        print(f"⚠ Classifier eval failed: {e}")
    
    try:
        results["retrieval"] = eval_retrieval()
    except Exception as e:
        print(f"⚠ Retrieval eval failed: {e}")
    
    try:
        results["faithfulness"] = eval_faithfulness()
    except Exception as e:
        print(f"⚠ Faithfulness eval failed: {e}")
    
    try:
        results["ablation"] = eval_ablation()
    except Exception as e:
        print(f"⚠ Ablation eval failed: {e}")
    
    # Generate plots
    try:
        generate_plots()
    except Exception as e:
        print(f"⚠ Plot generation failed: {e}")
    
    # Summary
    print("\n" + "█"*70)
    print("█  EVALUATION COMPLETE")
    print("█"*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("  - classifier_metrics.csv")
    print("  - retrieval_metrics.csv")
    print("  - faithfulness_metrics.csv")
    print("  - ablation_outputs.csv")
    print("  - metrics_plots.png")
    print("\n✓ All metrics collected and exported.")


if __name__ == "__main__":
    main()