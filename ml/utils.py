# ml/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_zeros(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replace 0 with NaN in specified columns indicating missing values."""
    df = df.copy()
    df[columns] = df[columns].replace(0, np.nan)
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split the DataFrame into train and test sets."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def evaluate_metrics(y_true, y_pred) -> dict:
    """Compute accuracy, precision, recall, and F1 score (weighted average)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
