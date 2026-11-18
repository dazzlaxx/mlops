"""Этап evaluate: метрики на тесте, запись metrics.json и данных для графика.

Если R2 ниже порога evaluate.min_r2, этап завершается с ошибкой: так модель,
которая хуже договорённого минимума, не попадает в сданную работу.
Метрики дописываются в тот же запуск MLflow, что открыл этап train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import tracking
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("evaluate")

FEATURES = Path("data/features/test.csv")
MODEL = Path("models/model.joblib")
METRICS_OUT = Path("reports/metrics.json")
PLOT_OUT = Path("reports/plots/pred_vs_true.csv")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    min_r2 = float(params["evaluate"]["min_r2"])

    bundle = joblib.load(MODEL)
    model, feature_names = bundle["model"], bundle["features"]

    frame = pd.read_csv(FEATURES)
    x_test = frame[feature_names]
    y_test = frame[target]
    y_pred = model.predict(x_test)

    metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
        "n_test": int(len(frame)),
    }
    ensure_parent(METRICS_OUT).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    sample = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).sample(
        n=min(500, len(frame)), random_state=int(params["train"]["seed"])
    )
    sample.to_csv(ensure_parent(PLOT_OUT), index=False)

    # Метрики пишутся до проверки порога: слабый запуск тоже виден в MLflow.
    tracking.log_metrics({k: v for k, v in metrics.items() if k != "n_test"})
    log.info("метрики: %s", metrics)
    if metrics["r2"] < min_r2:
        log.error("R2 %.4f ниже порога %.2f", metrics["r2"], min_r2)
        sys.exit(1)


if __name__ == "__main__":
    main()
