# Self-Healing MLOps Retraining Pipeline

An enterprise-grade production MLOps system that detects statistical feature drift on live unlabelled financial transaction data using two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI). Features automated background continuous retraining, Champion vs. Challenger model tournament evaluation, and atomic zero-downtime hot-swapping served via FastAPI and containerized in Docker.

## Architecture

```
[ LIVE INFERENCE LAYER ]
  Incoming Transaction ──► FastAPI (server.py) ──► Champion Model (Active)
                                 │
                                 ▼
                     Logs inputs to monitoring buffer
                                 │
═════════════════════════════════╪══════════════════════════════════════════════════
                                 │
[ THE SELF-HEALING ENGINE ]      ▼
  monitor.py  ──────────► Calculates Statistical Drift (KS-Test & PSI)
                          Compares Live Traffic vs. Baseline Training Data
                                 │
                                 ▼ (Statistical Drift Detected: p < 0.05)
  retrain.py  ──────────► Automatically ingests fresh transaction batch
                          Trains new Challenger Model
                                 │
                                 ▼
  evaluator.py ─────────► Champion vs. Challenger Tournament
                          Compares ROC-AUC, Recall, and Precision
                                 │
                                 ▼ (Challenger Wins: Recall > Champion + 2%)
  Hot-Swap Registry ────► Instructs server.py to reload weights in RAM
                          ZERO DOWNTIME. Model updated live!
```

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Drift Detection | Two-Sample Kolmogorov-Smirnov Test (`scipy.stats`), PSI |
| Model Core | PyTorch / Scikit-Learn Classifiers |
| Evaluation Protocol | Champion vs. Challenger Promotion Tournament |
| Serving & Deployment | FastAPI, Pydantic v2, Atomic In-Memory State Swapping |
| Containerization | Docker |
| Language | Python 3.11+ |

## Key Engineering Highlights

- **Unsupervised Drift Detection:** Detects distribution changes without ground-truth labels, solving the real-world Ground-Truth Lag problem.
- **Champion vs. Challenger Gate:** Eliminates regression risks by strictly gating automated deployments on multi-metric holdout comparisons.
- **Atomic Zero-Downtime Hot-Swapping:** Reassigns model pointers in RAM via FastAPI hooks in < 1 millisecond without restarting containers or dropping active requests.
- **Full Reproducibility:** Clean separation of data pipelines, monitoring services, and evaluation harnesses.

## Getting Started

```bash
git clone https://github.com/Lucas-Maingi/self-healing-mlops-pipeline.git
cd self-healing-mlops-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
python data_pipeline.py
python monitor.py
```

## License

MIT