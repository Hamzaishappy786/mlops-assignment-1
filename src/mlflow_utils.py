import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mlflow.models import infer_signature

# Get absolute path to avoid issues
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
mlruns_path = os.path.join(project_root, "mlruns_individual")
results_path = os.path.join(project_root, "results")

print(f"Project root: {project_root}")
print(f"MLruns path: {mlruns_path}")

# 📂 Ensure results dir exists
os.makedirs(results_path, exist_ok=True)

# 📊 Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_and_log(model, params, model_name):
    """Train, evaluate, and log model + metrics + confusion matrix to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log params + metrics
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        cm_path = os.path.join(results_path, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Log artifacts (plots + model)
        mlflow.log_artifact(cm_path)

        # Log model with signature and input example
        input_example = X_test[:5]
        signature = infer_signature(X_train[:10], model.predict(X_train[:10]))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature,
        )

        print(f"✅ {model_name} logged with Accuracy={acc:.3f}, F1={f1:.3f}")

# ⚡ Define Models
models = {
    "Logistic_Regression": (LogisticRegression(max_iter=200), {"max_iter": 200}),
    "SVM": (SVC(kernel="linear", C=1), {"kernel": "linear", "C": 1}),
    "KNN": (KNeighborsClassifier(n_neighbors=3), {"n_neighbors": 3})
}

# Set tracking URI with absolute path
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("mlruns_individual")

# 🚀 Train + Log each model
for model_name, (model, params) in models.items(): 
    evaluate_and_log(model, params, model_name)

print(f"\n📊 All models logged to: {mlruns_path}")
print(f"🌐 To view UI: mlflow ui --backend-store-uri \"file:///{mlruns_path.replace(os.sep, '/')}\" --host 127.0.0.1 --port 5000")

# Check if files were created
if os.path.exists(mlruns_path):
    print(f"✅ MLruns directory created at: {mlruns_path}")
else:
    print(f"❌ MLruns directory NOT created at: {mlruns_path}")