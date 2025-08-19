# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import DiabetesInput, DiabetesOutput
from app import service

app = FastAPI(
    title="Diabetes Prediction API",
    description="An API to predict diabetes from health data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "message": "Diabetes Prediction API is running.",
        "health": "/health",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=DiabetesOutput)
async def predict(data: DiabetesInput):
    """Predict diabetes given patient data."""
    pred_class, confidence = service.predict_diabetes(data)
    result_str = "Diabetic" if pred_class == 1 else "Not Diabetic"

    confidence = round(confidence, 2)
    return {"prediction": pred_class, "result": result_str, "confidence": confidence}

@app.get("/metrics")
async def metrics():
    """Return evaluation metrics from the model's test set."""
    import json
    with open("ml/metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics
