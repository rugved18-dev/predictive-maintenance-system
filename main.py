# main.py

# 1. Import necessary libraries
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# 2. Create a FastAPI app instance
app = FastAPI(title="Predictive Maintenance API")

# --- A MASTER'S INSIGHT ---
# We load the models ONCE when the API starts up.
# This is much more efficient than loading them for every single request.
print("Loading models and scalers...")
rul_model = joblib.load('xgb_model.joblib')
feature_scaler = joblib.load('feature_scaler.joblib')
anomaly_model = joblib.load('isolation_forest_model.joblib')
print("Models and scalers loaded successfully.")

# 3. Define the input data models using Pydantic
# This ensures that the data we receive is in the correct format.

class RULInput(BaseModel):
    # A sequence of 30 cycles, each with 24 features
    # Example: [[...24 features...], [...24 features...], ..., [...24 features...]]
    sequence: List[List[float]]

class AnomalyInput(BaseModel):
    # A single segment of 2048 vibration readings
    segment: List[float]


# 4. Define the API endpoints

# --- RUL PREDICTION ENDPOINT ---
@app.post("/predict_rul")
async def predict_rul(input_data: RULInput):
    """
    Takes a sequence of sensor data and predicts the Remaining Useful Life (RUL).
    """
    # 1. Convert input data to a NumPy array
    sequence_np = np.array(input_data.sequence)

    # 2. Check if the shape is correct (30, 24)
    if sequence_np.shape != (30, 24):
        return {"error": f"Invalid input shape. Expected (30, 24), but got {sequence_np.shape}"}

    # 3. Flatten the 3D sequence to 2D for the XGBoost model
    # Shape becomes (1, 30 * 24) which is (1, 720)
    flattened_sequence = sequence_np.reshape(1, -1)

    # 4. Make a prediction
    predicted_rul = rul_model.predict(flattened_sequence)[0]

    # 5. Return the result
    return {"predicted_rul": round(float(predicted_rul), 2)}


# --- ANOMALY DETECTION ENDPOINT ---
@app.post("/detect_anomaly")
async def detect_anomaly(input_data: AnomalyInput):
    """
    Takes a segment of vibration data and returns an anomaly score.
    """
    # 1. Convert input list to a NumPy array
    segment_np = np.array(input_data.segment)
    
    # 2. Check if the shape is correct (2048,)
    if segment_np.shape != (2048,):
        return {"error": f"Invalid input shape. Expected (2048,), but got {segment_np.shape}"}
        
    # 3. Reshape for the model, which expects a 2D array of samples
    # Shape becomes (1, 2048)
    reshaped_segment = segment_np.reshape(1, -1)

    # 4. Get the anomaly score using decision_function
    # Lower scores are more anomalous
    anomaly_score = anomaly_model.decision_function(reshaped_segment)[0]
    
    # 5. Determine the status based on the score (a common threshold is 0)
    status = "Anomaly Detected" if anomaly_score < 0 else "Normal"

    # 6. Return the results
    return {
        "anomaly_score": round(float(anomaly_score), 4),
        "status": status
    }

# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Predictive Maintenance API!"}