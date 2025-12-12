import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, "models")
mlruns_dir = os.path.join(project_root, "mlruns_individual")

# Set tracking URI
mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")

# Load data for signature
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model
print("Loading best model from .pkl file...")
best_model = joblib.load(os.path.join(models_dir, "logreg_model_95.pkl"))

# Register the model in MLflow Model Registry with more details
with mlflow.start_run(run_name="register_best_model_enhanced"):
    # Log parameters
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": 200,
        "test_size": 0.95,
        "train_size": 0.05,
        "dataset": "iris",
        "features": 4,
        "classes": 3
    })
    
    # Get model performance
    y_pred = best_model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    
    # Log model with signature and description
    input_example = X_test[:5]
    signature = infer_signature(X_train[:10], best_model.predict(X_train[:10]))
    
    # Register the model with description
    model_version = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        registered_model_name="iris_classifier"
    )
    
    # Add model description
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.update_registered_model(
            name="iris_classifier",
            description="Best performing model for Iris flower classification. Logistic Regression trained on 95% test split."
        )
        print("✅ Model description added")
    except Exception as e:
        print(f"Could not add description: {e}")
    
    print(f"✅ Model registered as 'iris_classifier'")
    print(f"Model URI: {model_version.model_uri}")
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")

# Test loading the registered model
print("\nTesting registered model...")
try:
    registered_model = mlflow.pyfunc.load_model("models:/iris_classifier/1")
    predictions = registered_model.predict(X_test[:5])
    print(f"✅ Registered model loaded successfully")
    print(f"Predictions: {predictions}")
except Exception as e:
    print(f"Error loading registered model: {e}")

print("\n" + "="*50)
print("To view in MLflow UI:")
print("1. Start UI: mlflow ui --backend-store-uri \"file:///C:/Users/Home/PycharmProjects/Basic%20projects/mlops-assignment-1/mlruns_individual\" --host 127.0.0.1 --port 5000")
print("2. Go to Models tab")
print("3. Click on 'iris_classifier'")
print("4. You should see version 1 with description and metrics")
# model enhancement