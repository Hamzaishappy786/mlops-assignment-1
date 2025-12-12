from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()
class ModelInput(BaseModel):
    feature1: float
    feature2: float

model_path = "models/model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f: model = pickle.load(f)
else: model = None

@app.get("/")
def home(): return {"message": "MLOps Assignment API is Running!"}

@app.get("/health")
def health_check():
    if model: return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
def predict(input_data: ModelInput):
    if not model: raise HTTPException(status_code=500, detail="Model not found")
    data = pd.DataFrame([[input_data.feature1, input_data.feature2]], columns=['feature1', 'feature2'])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}