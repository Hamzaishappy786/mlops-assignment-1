import os
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

def test_data_exists(): assert os.path.exists("data/dataset.csv"), "Dataset file is missing"

def test_data_loading():
    df = pd.read_csv("data/dataset.csv")
    assert not df.empty, "Dataset is empty"
    assert df.shape[0] > 10, "Dataset has too few rows"
    expected_cols = ["feature1", "feature2", "target"]
    assert all(col in df.columns for col in expected_cols), "Missing required columns"

def test_model_training():
    X_dummy = [[1.0, 2.0], [2.0, 3.0]]
    y_dummy = [0, 1]

    model = LogisticRegression()
    model.fit(X_dummy, y_dummy)

    assert hasattr(model, "coef_"), "Model failed to train"