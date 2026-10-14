import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

MODELS_DIR = "models"

def build_pipeline() -> Pipeline:
    """
    Constructs a machine learning pipeline that includes preprocessing and model training.
    Preprocessing: One-hot encoding for categorical features.
    Model: Random Forest Classifier for binary classification of credit risk.
    """
    # Define categorical and numerical features
    categorical_features = ["checking_status", "credit_history", "purpose", "housing"]
    numerical_features = ["credit_amount", "duration", "age"]

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    # Create the pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

def train_and_save_model(data_path: str, model_filename: str):
    """
    Loads data, trains the Random Forest pipeline, and saves weights to models/.
    """

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\nTraining model on: '{data_path}'...")

    df = pd.read_csv(data_path)
    x = df.drop(columns=["class"])
    # Convert 'good'/'bad' to binary 1 (good risk) and 0 (bad risk)
    y = (df["class"] == "good").astype(int)

    pipeline = build_pipeline()
    pipeline.fit(x, y)

    save_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(pipeline, save_path)
    print(f"  [Model Saved]: '{save_path}' (Trained on {len(df)} records)")
    return save_path

if __name__ == "__main__":
    # Step 1: Train the initial Champion Model on baseline data
    print("--- 1. Training Initial Production Champion Model ---")
    train_and_save_model("data/baseline_train.csv", "champion_model.joblib")

    # Step 2: Simulate Retraining when Drift occurs (Train Challenger on drifted batch)
    print("\n--- 2. Automated Retraining Triggered: Training Challenger Model ---")
    train_and_save_model("data/production_drifted.csv", "challenger_model.joblib")

    print("\nRetraining module verified. Champion and Challenger models created.")
