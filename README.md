# 🌸 MLOps Assignment 1: Iris Classification with MLflow

This project demonstrates **end-to-end MLOps workflow** on the classic Iris dataset.  
We train Logistic Regression, SVM, and KNN classifiers, track experiments with **MLflow**, compare results, and register the best model.

---

## 📂 Project Structure

```bash
mlops-assignment-1/
├── data/
├── models/                    # Saved models (*.pkl)
│   ├── logreg_model_95.pkl
│   ├── svm_model_95.pkl
│   └── knn_model_95.pkl
├── mlruns/                    # MLflow runs (main tracking dir)
├── notebooks/                 # Jupyter notebooks (exploration)
├── results/                   # Metrics JSON + confusion matrices
├── src/                       # Source codes
│   ├── Comparison_of_all_models.py
│   ├── mlflow_utils.py        # Core MLflow logging (95/5 split)
│   ├── train_logistic_regression.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── train_.py              # Stress test (95/5 split)
│   └── register_model_enhanced.py
└── README.md
```

---

## 🌱 Dataset

- **Source:** scikit-learn `load_iris()`
- **Features (4):** sepal length, sepal width, petal length, petal width  
- **Classes (3):** setosa, versicolor, virginica  
- **Size:** 150 samples

---

## 🤖 Models & Rationale

- **Logistic Regression** – simple baseline, interpretable.  
- **SVM (linear kernel)** – robust with high-dimensional separation.  
- **KNN** – non-parametric, leverages neighborhood similarity.

---

## ⚡ How to Run

### 1. Create virtual environment & install dependencies
```
python -m venv .venv
.venv\Scripts\activate     # (Windows)
pip install -r requirements.txt
```

### 2. Train models (95/5 split stress test)
```
python src/train_logistic_regression.py
python src/train_svm.py
python src/train_knn.py
```

### 3. Run combined trainer (95/5 split with MLflow logging)
```
python src/mlflow_utils.py
```

### 4. Start MLflow UI
From repo root:
```
mlflow ui --backend-store-uri file:///C:/Users/Home/PycharmProjects/Basic%20projects/mlops-assignment-1/mlruns_individual
Open http://127.0.0.1:5000
```
Runs are stored under Experiment: **iris_models_individual**

Artifacts (confusion matrices, JSON) are in results/
Saved models are in models/

## 📊 MLflow Tracking

Each run logs:
Parameters (e.g., hyperparameters)
Metrics: accuracy, precision, recall, F1
Artifacts: confusion matrix plots, metrics JSON
Serialized model (.pkl)

Compare runs:

Tick checkboxes in MLflow UI → Compare → view side-by-side metrics & charts.

| Model               | Accuracy | Precision | Recall | F1    |
| ------------------- | -------- | --------- | ------ | ----- |
| Logistic Regression | 0.741    | 0.855     | 0.741  | 0.697 |
| SVM (Linear)        | 0.944    | 0.952     | 0.944  | 0.944 |
| KNN                 | 0.783    | 0.869     | 0.783  | 0.758 |

📦 Model Registration

Best model: SVM (almost highest in every metric).

Register with:
```
python src/register_model_enhanced.py
```
Registered under name: iris_classifier

We can view versions in MLflow UI → Models tab → currently have 2 versions

🖼️ Screenshots (placeholders)

Expiremnt runs in MLFlow:
<img width="1920" height="610" alt="image" src="https://github.com/user-attachments/assets/00ddd28d-e91b-4886-af63-07c5ac362f11" />
 
Total runs in iris_models_individual:
<img width="1688" height="499" alt="image" src="https://github.com/user-attachments/assets/019f0ea8-7abe-4fee-864d-b679162100f8" />

Metric plots (accuracy/precision/recall/F1)
<img width="1674" height="359" alt="image" src="https://github.com/user-attachments/assets/9c372e4a-a973-4657-9c8c-360f0347f808" />

<img width="1920" height="841" alt="image" src="https://github.com/user-attachments/assets/8e1467fc-184c-4ea4-825f-0f630731c97d" />

 

Confusion matrices:
1. KNN:
<img width="500" height="400" alt="knn_confusion_95" src="https://github.com/user-attachments/assets/8da4d0fd-ca52-489b-92d6-083b808ebecb" />

2. SVM:
<img width="500" height="400" alt="svm_confusion_95" src="https://github.com/user-attachments/assets/e07dab46-4c91-4fb6-b620-bdadb98c05f2" />

3. Logistic Regression:
<img width="500" height="400" alt="logreg_confusion_95" src="https://github.com/user-attachments/assets/9f165813-47ee-4286-8b0e-4368832c7a5c" />


Models tab showing iris_classifier versions(currently 2):
<img width="1920" height="716" alt="image" src="https://github.com/user-attachments/assets/569ea150-13bf-41cb-bbbb-e7e1e4688173" />
 

🛠️ Repro Tips / Common Issues

MLflow UI shows “Failed to fetch” → Start from repo root & use absolute file:/// path.
Runs don’t appear → Ensure mlflow.set_tracking_uri() points to same store.
Windows paths with spaces → Use %20 in file URI (see command above).

### 📜 License & Acknowledgments

Dataset: scikit-learn Iris
This repo is for educational purposes (MLOps assignment#1).
