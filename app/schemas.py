# app/schemas.py

from pydantic import BaseModel

# Input schema for prediction
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Output schema for prediction response
class DiabetesOutput(BaseModel):
    prediction: int
    result: str
    confidence: float
