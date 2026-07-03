"""
Calculate Category Accuracy and Risk Flag Accuracy for the Full System
Compares model predictions against expected values from test scenarios.

Run: python eval_system_accuracy.py
"""
import csv
import os
import pandas as pd
from src.graph import run
from evaluation.test_prompts import SCENARIOS


def main():
    print("\n" + "="*70)
    print("EVALUATING SYSTEM ACCURACY")
    print("="*70)
    
    cat_correct = 0
    risk_correct = 0
    total = 0
    
    results = []
    
    for i, (text, expected_cat, expected_risk, is_abuse) in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Running: {text[:60]}...")
        
        result = run(text)
        
        predicted_cat = result.get("category", "unknown")
        predicted_risk = result.get("risk", "unknown")
        
        cat_match = predicted_cat == expected_cat
        risk_match = predicted_risk == expected_risk
        
        cat_correct += int(cat_match)
        risk_correct += int(risk_match)
        total += 1
        
        results.append({
            "Scenario": i,
            "Input": text[:50],
            "Expected Category": expected_cat,
            "Predicted Category": predicted_cat,
            "Category Match": "✓" if cat_match else "✗",
            "Expected Risk": expected_risk,
            "Predicted Risk": predicted_risk,
            "Risk Match": "✓" if risk_match else "✗",
        })
        
        print(f"  Category: {expected_cat} → {predicted_cat} {'✓' if cat_match else '✗'}")
        print(f"  Risk:     {expected_risk} → {predicted_risk} {'✓' if risk_match else '✗'}")
    
    # Save results to CSV
    results_dir = "evaluation/results"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "system_accuracy.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Calculate metrics
    cat_accuracy = cat_correct / total
    risk_accuracy = risk_correct / total
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n📊 Category Accuracy:  {cat_accuracy:.1%} ({cat_correct}/{total})")
    print(f"📊 Risk Flag Accuracy: {risk_accuracy:.1%} ({risk_correct}/{total})")
    
    # Save metrics to CSV
    metrics_data = {
        "Metric": ["Category Accuracy", "Risk Flag Accuracy"],
        "Score": [cat_accuracy, risk_accuracy],
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_dir, "system_accuracy_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"\n✓ Detailed results saved to: {csv_path}")
    print(f"✓ Metrics summary saved to: {metrics_path}")
    
    # Breakdown by category
    print("\n" + "="*70)
    print("CATEGORY BREAKDOWN")
    print("="*70)
    
    categories = df["Expected Category"].unique()
    for cat in categories:
        cat_data = df[df["Expected Category"] == cat]
        cat_acc = (cat_data["Category Match"] == "✓").sum() / len(cat_data)
        risk_acc = (cat_data["Risk Match"] == "✓").sum() / len(cat_data)
        print(f"\n{cat.upper()}:")
        print(f"  Samples: {len(cat_data)}")
        print(f"  Category Accuracy: {cat_acc:.1%}")
        print(f"  Risk Accuracy: {risk_acc:.1%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
