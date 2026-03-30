from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

app = FastAPI(title="Heart Stroke Prediction API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the model and scaler
model = None
scaler = None
model_accuracy = 0.0
dataset_shape = (0, 0)
dataset_averages = {}

@app.on_event("startup")
def load_and_train_model():
    global model, scaler, model_accuracy, dataset_shape, dataset_averages
    
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "heart.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    dataset_shape = df.shape
    
    # Calculate dataset averages for visualization
    try:
        avg_high = df[df["target"] == 1].mean().to_dict()
        avg_low = df[df["target"] == 0].mean().to_dict()
        dataset_averages = {"high_risk": avg_high, "low_risk": avg_low}
    except Exception as e:
        print(f"Failed to calculate averages: {e}")
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Initialize scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train the RandomForest model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully! Accuracy: {model_accuracy*100:.2f}%")


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/stats")
def get_stats():
    """Return model overview stats for the dashboard"""
    return {
        "dataset_size": dataset_shape[0],
        "features_count": max(0, dataset_shape[1] - 1),
        "model_accuracy": model_accuracy
    }

@app.get("/averages")
def get_averages():
    """Return mean feature values for healthy vs stroke patients"""
    return dataset_averages

@app.post("/predict")
def predict_risk(data: PatientData):
    """Predict risk based on patient features"""
    if model is None or scaler is None:
        return {"error": "Model not loaded"}
        
    # Convert input to numpy array in correct order
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    # prediction[0] is typically numpy.int64, convert to int for JSON
    
    return {
        "prediction": int(prediction[0]),
        "risk_level": "High" if int(prediction[0]) == 1 else "Low"
    }

@app.get("/")
def read_root():
    return {"message": "Heart Stroke Backend API is running."}
