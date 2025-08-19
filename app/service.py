# app/service.py

import numpy as np
import joblib

# Load the trained model pipeline at startup
model = joblib.load("ml/diabetes_model.pkl")

def predict_diabetes(features):
    # Convert input features into a numpy array of shape (1, 8) for the model
    data = np.array([[
        features.Pregnancies, 
        features.Glucose, 
        features.BloodPressure,
        features.SkinThickness, 
        features.Insulin, 
        features.BMI,
        features.DiabetesPedigreeFunction, 
        features.Age
    ]], dtype=float)
    # Replace 0 with NaN for columns where 0 is not a valid value
    # Columns indices: 2 = BloodPressure, 3 = SkinThickness, 4 = Insulin, 5 = BMI
    for idx in [2, 3, 4, 5]:
        if data[0, idx] == 0:
            data[0, idx] = np.nan
    # Make prediction using the loaded pipeline
    pred_class = int(model.predict(data)[0])
    # Calculate confidence of the prediction
    if hasattr(model, "predict_proba"):
        # If probability prediction is available, use it
        proba = model.predict_proba(data)[0]
        confidence = proba[pred_class]
    elif hasattr(model, "decision_function"):
        # If only decision function is available, convert to probability via sigmoid
        score = model.decision_function(data)[0]
        prob1 = 1 / (1 + np.exp(-score))
        confidence = prob1 if pred_class == 1 else 1 - prob1
    else:
        # Fallback
        confidence = 1.0
    return pred_class, float(confidence)
