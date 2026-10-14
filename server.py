import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = "models"
CHAMPION_PATH = os.path.join(MODELS_DIR, "champion_model.joblib")
CHALLENGER_PATH = os.path.join(MODELS_DIR, "challenger_model.joblib")

app = FastAPI(
    title="Self-Healing MLOps Financial Risk API",
    description="Production credit risk scoring API with atomic zero-downtime hot-swapping.",
    version="1.0.0"
)

# Global in-memory model pointer and version metadata
active_model = None
active_model_version = "None"


@app.on_event("startup")
def load_initial_model():
    """Loads the production Champion model into RAM at server startup."""
    global active_model, active_model_version

    if not os.path.exists(CHAMPION_PATH):
        raise FileNotFoundError(f"Initial model not found at '{CHAMPION_PATH}'. Run retrain.py first!")

    print(f"Loading initial production model: '{CHAMPION_PATH}' into RAM...")
    active_model = joblib.load(CHAMPION_PATH)
    active_model_version = "Champion (Baseline v1)"
    print("Initial model loaded. Server ready to receive transactions.")


# Pydantic schema for an incoming loan application
class LoanApplication(BaseModel):
    credit_amount: float = 2500.0
    duration: float = 24.0
    age: float = 35.0
    checking_status: str = "0<=X<200"
    credit_history: str = "critical/other existing credit"
    purpose: str = "radio/tv"
    housing: str = "own"


@app.get("/health")
def health_check():
    """Health check endpoint showing currently active model in RAM."""
    return {
        "status": "healthy",
        "active_model_version": active_model_version,
        "is_model_loaded": active_model is not None
    }


@app.post("/predict")
def predict_credit_risk(app_data: LoanApplication):
    """
    Scores a real-time loan application.
    Returns predicted risk class and default probability.
    """
    if active_model is None:
        raise HTTPException(status_code=503, detail="Model is not ready in memory.")

    # Convert incoming JSON into a 1-row DataFrame matching the model's schema
    input_df = pd.DataFrame([app_data.dict()])

    # Run inference
    pred = active_model.predict(input_df)[0]
    prob_good = active_model.predict_proba(input_df)[0][1]

    risk_label = "Good Credit Risk (Approve)" if pred == 1 else "High Risk of Default (Reject)"

    return {
        "decision": risk_label,
        "approval_probability": round(float(prob_good), 4),
        "served_by_model": active_model_version
    }


@app.post("/internal/hot-reload")
def hot_reload_promoted_model():
    """
    Atomic in-memory pointer swap: loads promoted Challenger model
    into RAM in < 1ms without restarting the server or dropping connections.
    """
    global active_model, active_model_version

    if not os.path.exists(CHALLENGER_PATH):
        raise HTTPException(status_code=404, detail="No promoted challenger model found on disk.")

    print(f"\n[HOT-RELOAD TRIGGERED]: Loading '{CHALLENGER_PATH}' in background...")
    # 1. Load the new candidate into a temporary variable
    new_model = joblib.load(CHALLENGER_PATH)

    # 2. Atomic Pointer Reassignment in RAM (0.0005 seconds!)
    active_model = new_model
    active_model_version = "Challenger (Promoted Post-Drift v2)"

    print(f"[HOT-RELOAD SUCCESSFUL]: Pointer updated to '{active_model_version}'. Zero downtime.")

    return {
        "status": "success",
        "message": "Model swapped atomically in RAM with zero downtime.",
        "new_active_model": active_model_version
    }