import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

MODELS_DIR = "models"
CHAMPION_PATH = os.path.join(MODELS_DIR, "champion_model.joblib")
CHALLENGER_PATH = os.path.join(MODELS_DIR, "challenger_model.joblib")

def load_held_out_test_set(data_path: str = "data/production_drifted.csv"):
    """
    Splits the drifted production batch into a training split (70%)
    and a strictly held-out test split (30%) to judge the tournament fairly.
    """

    df = pd.read_csv(data_path)
    x = df.drop(columns=["class"])
    y = (df["class"] == "good").astype(int)

    # 30% unseen holdout set
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    return x_test, y_test

def evaluate_model(model_pipeline, x_test, y_test) -> dict:
    """Calculates standard classification metrics on unseen data."""
    preds = model_pipeline.predict(x_test)
    probs = model_pipeline.predict_proba(x_test)[:, 1]

    return{
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4)
    }

def run_promotion_tournament(recall_margin: float = 0.02) -> dict:
    """
    Compares Champion vs Challenger on unseen test data.
    Promotes Challenger ONLY if Challenger Recall exceeds Champion by recall_margin (+2%).
    """

    if not os.path.exists(CHAMPION_PATH):
        raise FileNotFoundError(f"Champion model not found at: {CHAMPION_PATH}")
    if not os.path.exists(CHALLENGER_PATH):
        raise FileNotFoundError(f"Challenger model not found at: {CHALLENGER_PATH}")

    x_test, y_test = load_held_out_test_set()
    print(f"Tournament underway on {len(x_test)} unseen test records...")

    champion = joblib.load(CHAMPION_PATH)
    challenger = joblib.load(CHALLENGER_PATH)

    champ_metrics = evaluate_model(champion, x_test, y_test)
    chall_metrics = evaluate_model(challenger, x_test, y_test)

    # Promotion Gate: Did the Challenger beat the Champion by at least 2% Recall?
    recall_diff = chall_metrics["recall"] - champ_metrics["recall"]
    promoted = recall_diff >= recall_margin

    print("\n--- TOURNAMENT SCORECARD ---")
    print(f"  Metric       | Champion (Active) | Challenger (Candidate)")
    print(f"  Accuracy     | {champ_metrics['accuracy']:.4f}            | {chall_metrics['accuracy']:.4f}")
    print(f"  Recall       | {champ_metrics['recall']:.4f}            | {chall_metrics['recall']:.4f} (Diff: {recall_diff:+.4f})")
    print(f"  Precision    | {champ_metrics['precision']:.4f}            | {chall_metrics['precision']:.4f}")
    print(f"  ROC-AUC      | {champ_metrics['roc_auc']:.4f}            | {chall_metrics['roc_auc']:.4f}")
    if promoted:
        print(f"\n[DECISION]: CHALLENGER PROMOTED! (Beat Champion by {recall_diff*100:.2f}% Recall)")
    else:
        print(f"\n[DECISION]: CHALLENGER REJECTED. (Failed to beat Champion by required margin)")
    return {
        "promoted": promoted,
        "champion_metrics": champ_metrics,
        "challenger_metrics": chall_metrics
    }
if __name__ == "__main__":
    result = run_promotion_tournament(recall_margin=0.02)