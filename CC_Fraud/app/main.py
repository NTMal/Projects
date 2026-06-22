"""
main.py — FastAPI application serving fraud detection predictions.

This is the entry point for the Docker container.
When the container starts, it:
  1. Loads the pre-trained models from disk
  2. Starts a web server on port 8080
  3. Listens for prediction requests

Endpoints:
  GET  /health                → health check (used by Cloud Run to verify container is alive)
  GET  /                      → basic info about the API
  POST /predict/fraud         → supervised Random Forest prediction
  POST /predict/anomaly       → unsupervised Isolation Forest prediction
  POST /predict/combined      → both models, single request

Usage (once container is running):
  curl -X POST http://localhost:8080/predict/fraud \
    -H "Content-Type: application/json" \
    -d '{"V1": -1.35, "V2": -0.07, ..., "Amount": 149.62}'
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load models at startup
# We load models ONCE when the container starts, not on every request.
# This is important for performance as loading a Random Forest takes ~1 second.

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models():
    """Load all model artifacts from disk. Called once at startup."""
    try:
        rf    = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
        iso   = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        print("✓ Models loaded successfully")
        return rf, iso, scaler
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model file not found: {e}. "
            "Have you run train.py to generate the .joblib files?"
        )

# Load at module import time (i.e. when container starts)
rf_model, iso_model, scaler = load_models()

# FastAPI app 
# FastAPI automatically generates interactive docs at /docs
# Visit http://localhost:8080/docs once the container is running

app = FastAPI(
    title="CC Fraud Detection API",
    description=(
        "Serves two fraud detection models trained on the Kaggle Credit Card Fraud dataset:\n"
        "- **Random Forest** (supervised): trained with fraud labels, outputs fraud probability\n"
        "- **Isolation Forest** (unsupervised): trained without labels, outputs anomaly score"
    ),
    version="1.0.0",
)


# Input schema 
# Pydantic BaseModel validates incoming JSON automatically.
# If a required field is missing or the wrong type, FastAPI returns a 422 error
# with a clear message: no manual validation code needed.

class Transaction(BaseModel):
    """
    A single credit card transaction.
    V1 to V28 are PCA-transformed features from the Kaggle dataset.
    Amount is the transaction value in the original currency.
    Time is NOT included — we drop it during preprocessing.
    """
    # V1 to V28: PCA components (already anonymised in the dataset)
    V1:  float = Field(..., example=-1.3598071336738)
    V2:  float = Field(..., example=-0.0727811733098497)
    V3:  float = Field(..., example=2.53634673796914)
    V4:  float = Field(..., example=1.37815522427443)
    V5:  float = Field(..., example=-0.338320769942518)
    V6:  float = Field(..., example=0.462387777762292)
    V7:  float = Field(..., example=0.239598554061257)
    V8:  float = Field(..., example=0.0986979012610507)
    V9:  float = Field(..., example=0.363786969611213)
    V10: float = Field(..., example=0.0907941719789316)
    V11: float = Field(..., example=-0.551599533260813)
    V12: float = Field(..., example=-0.617800855762348)
    V13: float = Field(..., example=-0.991389847235408)
    V14: float = Field(..., example=-0.311169353699879)
    V15: float = Field(..., example=1.46817697209427)
    V16: float = Field(..., example=-0.470400525259478)
    V17: float = Field(..., example=0.207971241929242)
    V18: float = Field(..., example=0.0257905801985591)
    V19: float = Field(..., example=0.403992960255733)
    V20: float = Field(..., example=0.251412098239705)
    V21: float = Field(..., example=-0.018306777944153)
    V22: float = Field(..., example=0.277837575558899)
    V23: float = Field(..., example=-0.110473910188767)
    V24: float = Field(..., example=0.0669280749146731)
    V25: float = Field(..., example=0.128539358273528)
    V26: float = Field(..., example=-0.189114843888824)
    V27: float = Field(..., example=0.133558376740387)
    V28: float = Field(..., example=-0.0210530534538215)
    Amount: float = Field(..., example=149.62, description="Transaction amount in original currency")


# Preprocessing helper

def preprocess_transaction(transaction: Transaction) -> pd.DataFrame:
    """
    Convert a Transaction object into a preprocessed DataFrame
    ready for model inference.

    This mirrors the unified preprocessing in train.py:
      - No Time column (was never sent)
      - Scale Amount using the fitted scaler
      - Keep V1 to V28 as-is

    Returns a single-row DataFrame with 29 features.
    """
    # Convert Pydantic model to dict, then to DataFrame
    data = transaction.model_dump()
    df = pd.DataFrame([data])  # single-row DataFrame

    # Scale Amount using the SAME scaler fitted during training
    # This is critical — using a different scale would give wrong predictions
    df["Amount"] = scaler.transform(df[["Amount"]])

    # Ensure column order matches training data: V1, V2, ..., V28, Amount
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = df[feature_cols]

    return df


# Endpoints
@app.get("/")
def root():
    """Basic API info. Visit /docs for interactive documentation."""
    return {
        "name": "CC Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health check":         "GET  /health",
            "supervised prediction": "POST /predict/fraud",
            "anomaly detection":     "POST /predict/anomaly",
            "combined prediction":   "POST /predict/combined",
            "interactive docs":      "GET  /docs",
        },
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    GCP Cloud Run pings this to verify the container is alive and ready.
    Must return HTTP 200 for the service to receive traffic.
    """
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict/fraud")
def predict_fraud(transaction: Transaction):
    """
    Supervised fraud prediction using Random Forest.

    Returns:
      - fraud_probability : float [0–1], probability this transaction is fraud
      - prediction        : str, 'FRAUD' or 'LEGIT'
      - model             : str, which model was used
    """
    try:
        # Preprocess the incoming transaction
        X = preprocess_transaction(transaction)

        # predict_proba returns [[prob_legit, prob_fraud]]
        # We take index [0][1] = probability of class 1 (fraud) for first (only) row
        fraud_prob = rf_model.predict_proba(X)[0][1]

        # Threshold at 0.5: above = fraud, below = legit
        # In production you might tune this threshold to trade off precision vs recall
        prediction = "FRAUD" if fraud_prob >= 0.5 else "LEGIT"

        return {
            "prediction": prediction,
            "fraud_probability": round(float(fraud_prob), 4),
            "model": "RandomForest (supervised)",
        }

    except Exception as e:
        # Return a clear error message if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/anomaly")
def predict_anomaly(transaction: Transaction):
    """
    Unsupervised anomaly detection using Isolation Forest.

    Returns:
      - anomaly_score : float (more negative = more anomalous)
      - prediction    : str, 'ANOMALY' or 'NORMAL'
      - model         : str, which model was used

    Note on anomaly_score:
      - Isolation Forest returns a score via decision_function()
      - Negative scores = anomaly (fraud), positive scores = normal
      - The more negative, the more isolated (anomalous) the transaction
    """
    try:
        X = preprocess_transaction(transaction)

        # decision_function() returns the raw anomaly score
        # More negative = more anomalous = more likely to be fraud
        anomaly_score = iso_model.decision_function(X)[0]

        # predict() returns 1 (normal) or -1 (anomaly)
        raw_pred = iso_model.predict(X)[0]
        prediction = "ANOMALY" if raw_pred == -1 else "NORMAL"

        return {
            "prediction": prediction,
            "anomaly_score": round(float(anomaly_score), 4),
            "note": "More negative score = more anomalous",
            "model": "IsolationForest (unsupervised)",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/combined")
def predict_combined(transaction: Transaction):
    """
    Run both models on a single transaction and return both results.

    This mirrors real-world fraud systems where supervised and unsupervised
    models run in parallel — agreement between them increases confidence.

    Returns both model outputs plus a simple combined flag:
      - combined_alert: True if EITHER model flags the transaction
    """
    try:
        X = preprocess_transaction(transaction)

        # ── Random Forest ──
        fraud_prob = rf_model.predict_proba(X)[0][1]
        rf_prediction = "FRAUD" if fraud_prob >= 0.5 else "LEGIT"

        # ── Isolation Forest ──
        anomaly_score = iso_model.decision_function(X)[0]
        iso_raw = iso_model.predict(X)[0]
        iso_prediction = "ANOMALY" if iso_raw == -1 else "NORMAL"

        # Combined alert: flag if either model thinks it's suspicious
        combined_alert = rf_prediction == "FRAUD" or iso_prediction == "ANOMALY"

        return {
            "combined_alert": combined_alert,
            "random_forest": {
                "prediction": rf_prediction,
                "fraud_probability": round(float(fraud_prob), 4),
            },
            "isolation_forest": {
                "prediction": iso_prediction,
                "anomaly_score": round(float(anomaly_score), 4),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
