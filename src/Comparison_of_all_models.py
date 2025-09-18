import json
from pathlib import Path
import matplotlib.pyplot as plt

# Absolute project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
results_dir = PROJECT_ROOT / "results"

# List of model metrics files
metric_files = {
    "Logistic Regression": results_dir / "logreg_metrics.json",
    "SVM": results_dir / "svm_metrics.json",
    "KNN": results_dir / "knn_metrics.json",
}

comparison = {}

# Read each JSON and collect key metrics
for model_name, filepath in metric_files.items():
    if not filepath.exists():
        print(f"⚠️ Missing metrics file for {model_name}: {filepath}")
        continue

    with open(filepath, "r") as f:
        data = json.load(f)

    acc = data["accuracy"]
    # Grab macro average scores for precision, recall, f1
    precision = data["report"]["macro avg"]["precision"]
    recall = data["report"]["macro avg"]["recall"]
    f1 = data["report"]["macro avg"]["f1-score"]

    comparison[model_name] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

# Save combined JSON
output_path = results_dir / "comparison_metrics.json"
with open(output_path, "w") as f:
    json.dump(comparison, f, indent=4)

print(f"✅ Comparison metrics saved at {output_path}")

# ---- Plotting ----
if comparison:
    models = list(comparison.keys())
    accuracies = [comparison[m]["accuracy"] for m in models]
    precisions = [comparison[m]["precision"] for m in models]
    recalls = [comparison[m]["recall"] for m in models]
    f1s = [comparison[m]["f1_score"] for m in models]

    x = range(len(models))
    width = 0.2

    plt.figure(figsize=(8, 6))
    plt.bar([i - 1.5*width for i in x], accuracies, width, label="Accuracy")
    plt.bar([i - 0.5*width for i in x], precisions, width, label="Precision")
    plt.bar([i + 0.5*width for i in x], recalls, width, label="Recall")
    plt.bar([i + 1.5*width for i in x], f1s, width, label="F1-score")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Model Comparison on Iris Dataset")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()

    plot_path = results_dir / "comparison_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Comparison plot saved at {plot_path}")

    # ---- Line Chart for Accuracy ----
    plt.figure(figsize=(8, 5))
    plt.plot(models, accuracies, marker="o", linestyle="-", color="b", label="Accuracy")
    plt.ylim(0, 1.1)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison (Line Chart)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    line_plot_path = results_dir / "accuracy_line_chart.png"
    plt.savefig(line_plot_path)
    plt.close()

    print(f"📈 Accuracy line chart saved at {line_plot_path}")
    #comp