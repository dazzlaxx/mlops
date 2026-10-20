"""evaluate: метрики классификации."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("evaluate")

FEATURES = Path("data/features/test.csv")
MODEL = Path("models/model.joblib")
METRICS_OUT = Path("reports/metrics.json")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    min_accuracy = float(params["evaluate"]["min_accuracy"])

    bundle = joblib.load(MODEL)
    model, feature_names = bundle["model"], bundle["features"]

    frame = pd.read_csv(FEATURES)
    x_test = frame[feature_names]
    y_test = frame[target]
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "n_test": int(len(frame)),
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("метрики: %s", metrics)

    if metrics["accuracy"] < min_accuracy:
        log.error("accuracy = %.4f ниже порога %.2f", metrics["accuracy"], min_accuracy)
        sys.exit(1)


if __name__ == "__main__":
    main()
