import os
import pandas as pd
from scipy.stats import ks_2samp

# Numeric features to monitor for distribution shifts
MONITORED_FEATURES = ["credit_amount", "duration", "age"]


class DriftMonitor:
    """
    Unsupervised feature drift detector using the Two-Sample
    Kolmogorov-Smirnov (KS) test.
    """
    def __init__(self, baseline_path: str = "data/baseline_train.csv", p_threshold: float = 0.05):
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline file missing: {baseline_path}")

        self.baseline_df = pd.read_csv(baseline_path)
        self.p_threshold = p_threshold
        print(f"DriftMonitor initialized. Baseline loaded ({len(self.baseline_df)} records).")

    def check_drift(self, incoming_df: pd.DataFrame) -> dict:
        """
        Runs KS-test on each monitored feature comparing incoming data to baseline.
        Returns a detailed report and a boolean flag indicating if drift was detected.
        """
        drift_report = {}
        drift_detected = False

        for feature in MONITORED_FEATURES:
            baseline_vals = self.baseline_df[feature].dropna()
            incoming_vals = incoming_df[feature].dropna()

            # Two-Sample Kolmogorov-Smirnov Test
            stat, p_val = ks_2samp(baseline_vals, incoming_vals)

            # If p-value is below threshold (0.05), the distributions are statistically different
            feature_drifted = p_val < self.p_threshold

            if feature_drifted:
                drift_detected = True

            drift_report[feature] = {
                "ks_statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "drift_detected": feature_drifted
            }

        return {
            "drift_detected": drift_detected,
            "metrics": drift_report
        }


if __name__ == "__main__":
    monitor = DriftMonitor(baseline_path="data/baseline_train.csv")

    # Test 1: Normal Production Batch (Should NOT drift)
    print("\n--- Test 1: Monitoring Normal Production Batch ---")
    normal_df = pd.read_csv("data/production_normal.csv")
    report_normal = monitor.check_drift(normal_df)
    print(f"Overall Drift Detected: {report_normal['drift_detected']}")
    for feat, res in report_normal["metrics"].items():
        print(f"  {feat:15} | KS-Stat: {res['ks_statistic']:.4f} | p-val: {res['p_value']:.6f} | Drift: {res['drift_detected']}")

    # Test 2: Drifted Production Batch (Macro Inflation Shift -> MUST drift)
    print("\n--- Test 2: Monitoring Drifted Production Batch ---")
    drifted_df = pd.read_csv("data/production_drifted.csv")
    report_drifted = monitor.check_drift(drifted_df)
    print(f"Overall Drift Detected: {report_drifted['drift_detected']}")
    for feat, res in report_drifted["metrics"].items():
        print(f"  {feat:15} | KS-Stat: {res['ks_statistic']:.4f} | p-val: {res['p_value']:.6f} | Drift: {res['drift_detected']}")