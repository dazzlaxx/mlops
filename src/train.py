"""train: обучение модели."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.utils import ensure_parent, get_logger, load_params, set_seed

log = get_logger("train")
FEATURES = Path("data/features/train.csv")
MODEL_OUT = Path("models/model.joblib")


def make_model(tp: dict):
    name = tp["model"]
    seed = int(tp["seed"])
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(tp["n_estimators"]),
            max_depth=int(tp["max_depth"]),
            min_samples_leaf=int(tp["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(tp["n_estimators"]),
            max_depth=int(tp["max_depth"]),
            min_samples_leaf=int(tp["min_samples_leaf"]),
            learning_rate=float(tp["learning_rate"]),
            random_state=seed,
        )
    raise ValueError(f"неизвестная модель: {name}")


def main() -> None:
    params = load_params()
    tp = params["train"]
    target = params["data"]["target"]
    set_seed(int(tp["seed"]))

    frame = pd.read_csv(FEATURES)
    x_train = frame.drop(columns=[target])
    y_train = frame[target]

    model = make_model(tp)
    model.fit(x_train, y_train)

    bundle = {"model": model, "features": list(x_train.columns)}
    joblib.dump(bundle, ensure_parent(MODEL_OUT))
    log.info("обучено %s на %d строках", tp["model"], len(frame))


if __name__ == "__main__":
    main()
