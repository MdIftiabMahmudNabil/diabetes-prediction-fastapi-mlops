# Diabetes Prediction — FastAPI + scikit-learn + Streamlit

End-to-end assignment: train a classifier on the **Pima Indians Diabetes** dataset, serve it via an **async FastAPI** API, containerize with **Docker**, deploy on **Render**, and provide a **Streamlit**.

## Live URLs

- **Streamlit UI:** https://diabetes-prediction-fastapi-mlops-6qycjtwb8utorrnniio75v.streamlit.app/
- **FastAPI (Render):** https://diabetes-prediction-fastapi-mlops.onrender.com
  - Health: https://diabetes-prediction-fastapi-mlops.onrender.com/health
  - Docs (Swagger): https://diabetes-prediction-fastapi-mlops.onrender.com/docs

---

## Features

- ✅ Trains **two models** (Random Forest, KNN) and **selects best by weighted F1**
- ✅ Full **sklearn Pipeline** (median impute → standardize → classifier)
- ✅ Handles biologically impossible zeros (`BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) → `NaN` → **median imputation**
- ✅ Saves artifacts:
  - `ml/diabetes_model.pkl` (joblib)
  - `ml/metrics.json` (accuracy, precision, recall, f1)
- ✅ **Async FastAPI** endpoints:
  - `GET /health`
  - `POST /predict` (returns `prediction`, `result`, `confidence`)
  - `GET /metrics` (bonus: saved test metrics)
- ✅ **Dockerfile** and optional **docker-compose**
- ✅ **Render** deployment config (`render.yaml`)
- ✅ Two frontends:
  - Streamlit app with modern UI
  - Minimal HTML + JS page

---

## Dataset

- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Expected CSV path: **`data/diabetes.csv`**
- Required columns (9):  
  `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

> `Outcome` is the target (0 = not diabetic, 1 = diabetic).

---

## Project Structure

```text
diabetes-prediction/
├── ml/
│ ├── train.py # training entrypoint
│ ├── utils.py # helpers (load/clean/split/evaluate)
│ ├── metrics.json # test metrics (generated)
│ └── diabetes_model.pkl # trained pipeline (generated)
├── app/
│ ├── main.py # FastAPI app (endpoints + wiring)
│ ├── schemas.py # Pydantic models
│ ├── service.py # load model + predict
│ └── init.py
├── frontend/
│ ├── streamlit_app.py # Streamlit UI
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── .dockerignore      # Docker ignore rules
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```


---

## Quickstart (local)

```bash
> Requires **Python 3.11**.
```
### 1) Install deps

```bash
cd diabetes-prediction
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#  Usage

## 2) Put the dataset
Download from Kaggle and place the CSV at:
```bash
data/diabetes.csv
```

## 3) Train the model
```bash
python ml/train.py
# creates: ml/diabetes_model.pkl and ml/metrics.json
```
## 4) Run the API
Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Once running, test the endpoints:

- Health check: **http://localhost:8000/health**
- Interactive API docs (Swagger UI): **http://localhost:8000/docs**

## 5) Try a prediction (example)
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 3,
    "Glucose": 145,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 29
  }'
```
## Frontends
### Streamlit
 By default it targets your Render API; override with API_URL if needed.
```bash
# uses https://diabetes-prediction-fastapi-mlops.onrender.com by default
streamlit run frontend/streamlit_app.py

# point to local API instead
API_URL="http://localhost:8000" streamlit run frontend/streamlit_app.py
```
## API Endpoints

### 1. Health Check
**GET** `/health`

**Response**
```json
{
  "status": "healthy"
}
```

### 2. Diabetes Prediction
**POST** `/predict`

**Request Body**
```json
{
  "Pregnancies": 3,
  "Glucose": 145,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.35,
  "Age": 29
}
```
**Response**
```json
{
  "prediction": 0,
  "result": "Not Diabetic",
  "confidence": 0.87
}
```
- **prediction:** 0 (not diabetic) or 1 (diabetic)
- **confidence:** probability for the predicted class (0–1), rounded to 2 decimals

### 3. Accuracy Metrics Check
**GET** `/metrics`

**Response**
```json
{ "accuracy": 0.78, "precision": 0.79, "recall": 0.78, "f1": 0.78 }
```
## Training Details

- **Split:** `train_test_split` with `test_size=0.2`, `random_state=42`, `stratify=y`

- **Preprocess:**
  - Zeros → `NaN` for: `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
  - `SimpleImputer(strategy="median")`
  - `StandardScaler`

- **Models compared:**
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
  - `KNeighborsClassifier()`

- **Model selection:** Best model chosen by **weighted F1** score on the test set

- **Artifacts:**
  - `ml/diabetes_model.pkl` — full scikit-learn Pipeline saved via joblib
  - `ml/metrics.json` — metrics (accuracy, precision, recall, f1) rounded to 4 decimals

### Feature Descriptions

| Feature                      | Description                                                   | Values / Units          |
|-----------------------------|---------------------------------------------------------------|-------------------------|
| `Pregnancies`               | Number of times the patient has been pregnant                 | Integer (count)         |
| `Glucose`                   | Plasma glucose concentration                                  | Float (mg/dL)           |
| `BloodPressure`             | Diastolic blood pressure                                      | Float (mm Hg)           |
| `SkinThickness`             | Triceps skin fold thickness                                   | Float (mm)              |
| `Insulin`                   | 2-hour serum insulin                                          | Float (IU/mL)           |
| `BMI`                       | Body Mass Index                                               | Float (kg/m²)           |
| `DiabetesPedigreeFunction`  | Score estimating diabetes risk based on family history        | Float (unitless)        |
| `Age`                       | Patient age                                                   | Integer (years)         |

> **Note:** In preprocessing, biologically impossible zeros in `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing and imputed with the median.

## Docker Deployment

> **Note:** Make sure you’ve trained the model first so `ml/diabetes_model.pkl` exists, or use the build-time training variant of the Dockerfile.

### Build Docker Image
```bash
docker build -t diabetes-api .
```
### Run Docker Container
```bash
docker run -d -p 8000:8000 diabetes-api
# Test: http://localhost:8000/health
```
### Using Docker Compose
```bash
docker compose up -d
# Test: http://localhost:8000/health
```
## Cloud Deployment

This project uses a **Dockerfile** at the repo root and a pre-trained model committed to `ml/diabetes_model.pkl`.

### A. Push to GitHub
```bash
git add .
git commit -m "Deploy-ready: Docker + model artifact"
git branch -M main
git remote add origin https://github.com/<your-username>/diabetes-prediction-fastapi-mlops.git
git push -u origin main
```
### B) Create a Web Service on Render

1. Go to **Render → New → Web Service**  
2. **Connect** your GitHub repository  
3. **Environment/Language:** Docker *(auto-detected)*  
4. **Branch:** `main`  
5. **Root Directory:** *(leave blank — Dockerfile is at repo root)*  
6. **Instance Type:** **Free** *(sufficient for this demo)*  
7. **Environment Variables:** *none required*  
8. Click **Create Web Service**


### C) Post-create setting

In **Render → Settings → Health Check Path**, set: /health

### D) Verify & Test

1. Watch **Logs** until you see:
```bash
Uvicorn running on http://0.0.0.0:<PORT>
Application startup complete.
```
2. Open your live URL and test:

- `/health` → liveness check  
- `/metrics` → model test metrics (accuracy, precision, recall, f1)
- `/docs` → Swagger UI

### Other Platforms
The Docker container can be deployed to:
- **Heroku**
- **AWS ECS**
- **Google Cloud Run**
- **Azure Container Instances**

## Model Information
- **Algorithm:** Best of **Random Forest** or **KNN** (chosen by **weighted F1** on test set)  
- **Training Data:** Pima Indians Diabetes Dataset (`data/diabetes.csv`)  
- **Features (8):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`  
- **Output:** Binary classification  
  - `prediction`: `0` (Not Diabetic) or `1` (Diabetic)  
  - `result`: `"Not Diabetic"` / `"Diabetic"`  
  - `confidence`: probability for predicted class (0–1)  
- **Model File:** `ml/diabetes_model.pkl`  
- **Metrics File:** `ml/metrics.json` (accuracy, precision, recall, f1; 4 decimals)

## Development

### Adding New Features
1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests if applicable  
5. Submit a pull request

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.  
For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Pima Indians Diabetes Dataset (Kaggle)  
- FastAPI framework  
- scikit-learn library

## Contact
**Author:** Md Iftiab Mahmud Nabil  
**Repository:** diabetes-prediction-fastapi-mlops

---

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
