# frontend/streamlit_app.py

import streamlit as st
import requests

# Configure the API endpoint
API_URL = "http://localhost:8000"  # Change to your Render URL when deployed, e.g., "https://diabetes-prediction-api.onrender.com"

st.title("Diabetes Prediction App")
st.write("Enter the patient data below and click **Predict** to see if the model predicts diabetes.")

# Input fields for each feature
pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
glucose = st.number_input("Glucose", min_value=0, step=1, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1, value=0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, step=1, value=0)
insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, step=1, value=0)
bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0, step=1, value=0)

# When the Predict button is clicked
if st.button("Predict"):
    # Prepare the payload for the API
    data = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": int(age)
    }
    with st.spinner("Making prediction..."):
        try:
            # Call the FastAPI predict endpoint
            response = requests.post(f"{API_URL}/predict", json=data, timeout=5)
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
        else:
            if response.status_code == 200:
                result = response.json()
                # Extract result and confidence
                label = result["result"]
                confidence = result["confidence"] * 100
                # Display the result
                st.success(f"**Prediction:** {label} (Confidence: {confidence:.2f}%)")
            else:
                # API returned an error
                st.error(f"API error: {response.status_code} - {response.text}")
