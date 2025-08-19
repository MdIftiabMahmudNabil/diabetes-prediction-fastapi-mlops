# ml/train.py

import os
import json
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import utils


# Ensure the output directory exists
os.makedirs("ml", exist_ok=True)

# Load and preprocess the dataset
df = utils.load_data("data/diabetes.csv")
df = utils.clean_zeros(df, ["BloodPressure", "SkinThickness", "Insulin", "BMI"])
X_train, X_test, y_train, y_test = utils.split_data(df, test_size=0.2, random_state=42)

# classifiers to compare
candidates = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_model_name = None
best_f1 = -1.0
best_metrics = None

# Train and evaluate each classifier
for name, clf in candidates.items():
    # Build a pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = utils.evaluate_metrics(y_test, y_pred)
    # Print metrics for this model
    print(f"{name} -> Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    # Select the model with the highest F1 score
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_model = pipeline
        best_model_name = name
        best_metrics = metrics

# Save the best model and its metrics
if best_model is not None:
    # Save the entire pipeline
    dump(best_model, "ml/diabetes_model.pkl")
    # Round metrics to 4 decimal places and save to JSON
    best_metrics = {k: round(v, 4) for k, v in best_metrics.items()}
    with open("ml/metrics.json", "w") as f:
        json.dump(best_metrics, f)
    print(f"Best model: {best_model_name} (F1 = {best_f1:.4f})")
    print("Saved model to ml/diabetes_model.pkl and metrics to ml/metrics.json")
else:
    print("No model was trained or saved.")
