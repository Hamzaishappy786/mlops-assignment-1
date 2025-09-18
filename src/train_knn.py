import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

# Absolute project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use top-level models/ and results/
models_dir = PROJECT_ROOT / "models"
results_dir = PROJECT_ROOT / "results"

if not models_dir.exists():
    raise FileNotFoundError(f"Expected 'models/' folder at {models_dir}, but not found.")
if not results_dir.exists():
    raise FileNotFoundError(f"Expected 'results/' folder at {results_dir}, but not found.")

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.95, random_state=42
)

# Train model (default k=5)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Save metrics JSON
metrics_path = results_dir / "knn_metrics.json"
with open(metrics_path, "w") as f:
    json.dump({"accuracy": acc, "report": report}, f, indent=4)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(results_dir / "knn_confusion.png")
plt.close()

# Save model
model_path = models_dir / "knn_model.pkl"
joblib.dump(clf, model_path)

print(f"[KNN] Done → Accuracy: {acc:.4f}")
print(f"Model saved at: {model_path}")
print(f"Metrics saved at: {metrics_path}")