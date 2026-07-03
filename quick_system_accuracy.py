"""
Extract Category and Risk Flag Accuracy from Existing Ablation Outputs
Uses already-generated ablation_outputs.csv to calculate metrics quickly.

Run: python quick_system_accuracy.py
"""
import pandas as pd
import os

# Load ablation outputs
ablation_path = "evaluation/results/ablation_outputs.csv"

if not os.path.exists(ablation_path):
    print(f"❌ File not found: {ablation_path}")
    print("Run 'python -m evaluation.ablation' first")
    exit(1)

df = pd.read_csv(ablation_path)

print("\n" + "="*70)
print("SYSTEM ACCURACY FROM ABLATION STUDY")
print("="*70)

# Calculate metrics for system C (multi-agent)
cat_correct = (df['expected_category'] == df['C_category']).sum()
risk_correct = (df['expected_risk'] == df['C_risk']).sum()
total = len(df)

cat_accuracy = cat_correct / total
risk_accuracy = risk_correct / total

print(f"\n📊 Category Accuracy:  {cat_accuracy:.1%} ({cat_correct}/{total})")
print(f"📊 Risk Flag Accuracy: {risk_accuracy:.1%} ({risk_correct}/{total})")

# Save metrics
results_dir = "evaluation/results"
metrics_path = os.path.join(results_dir, "system_accuracy_metrics.csv")
metrics_data = pd.DataFrame({
    "Metric": ["Category Accuracy", "Risk Flag Accuracy"],
    "Score": [cat_accuracy, risk_accuracy],
})
metrics_data.to_csv(metrics_path, index=False)

print(f"\n✓ Metrics saved to: {metrics_path}")

# Breakdown by category
print("\n" + "="*70)
print("BREAKDOWN BY CATEGORY")
print("="*70)

categories = df['expected_category'].unique()
for cat in sorted(categories):
    cat_data = df[df['expected_category'] == cat]
    cat_acc = (cat_data['expected_category'] == cat_data['C_category']).sum() / len(cat_data)
    risk_acc = (cat_data['expected_risk'] == cat_data['C_risk']).sum() / len(cat_data)
    print(f"\n{cat.upper()}:")
    print(f"  Samples: {len(cat_data)}")
    print(f"  Category Accuracy: {cat_acc:.1%}")
    print(f"  Risk Accuracy: {risk_acc:.1%}")

# Breakdown by risk level
print("\n" + "="*70)
print("BREAKDOWN BY RISK LEVEL")
print("="*70)

risks = df['expected_risk'].unique()
for risk in sorted(risks):
    risk_data = df[df['expected_risk'] == risk]
    cat_acc = (risk_data['expected_category'] == risk_data['C_category']).sum() / len(risk_data)
    risk_acc = (risk_data['expected_risk'] == risk_data['C_risk']).sum() / len(risk_data)
    print(f"\n{risk.upper()}:")
    print(f"  Samples: {len(risk_data)}")
    print(f"  Category Accuracy: {cat_acc:.1%}")
    print(f"  Risk Accuracy: {risk_acc:.1%}")

# Show misclassifications
print("\n" + "="*70)
print("MISCLASSIFICATIONS")
print("="*70)

cat_errors = df[df['expected_category'] != df['C_category']]
risk_errors = df[df['expected_risk'] != df['C_risk']]

if len(cat_errors) > 0:
    print(f"\n❌ {len(cat_errors)} Category Mismatches:")
    for idx, row in cat_errors.iterrows():
        print(f"  • {row['input'][:50]}...")
        print(f"    Expected: {row['expected_category']} → Got: {row['C_category']}")

if len(risk_errors) > 0:
    print(f"\n⚠️  {len(risk_errors)} Risk Flag Mismatches:")
    for idx, row in risk_errors.iterrows():
        print(f"  • {row['input'][:50]}...")
        print(f"    Expected: {row['expected_risk']} → Got: {row['C_risk']}")

print("\n" + "="*70)
