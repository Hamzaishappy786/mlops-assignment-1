import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Absolute project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
mlruns_dir = PROJECT_ROOT / "mlruns_individual"

# Force MLflow logs into top-level mlruns_individual/
mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
mlflow.set_experiment("iris_models_individual")

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

# Train and log with MLflow
with mlflow.start_run(run_name="KNN_95_5"):
    # Train model (default k=5)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log parameters and metrics
    mlflow.log_params({"n_neighbors": 5, "test_size": 0.95, "train_size": 0.05})
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - KNN (95% test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = results_dir / "knn_confusion_95.png"
    plt.savefig(cm_path)
    plt.close()

    # Log artifacts
    mlflow.log_artifact(str(cm_path))

    # Log model
    input_example = X_test[:5]
    signature = infer_signature(X_train[:10], clf.predict(X_train[:10]))
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="KNN",
        input_example=input_example,
        signature=signature,
    )

    # Save metrics JSON
    metrics_path = results_dir / "knn_metrics_95.json"
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "report": report}, f, indent=4)

    # Save model
    model_path = models_dir / "knn_model_95.pkl"
    joblib.dump(clf, model_path)

    print(f"[KNN] Done → Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Model saved at: {model_path}")
    print(f"Metrics saved at: {metrics_path}")