"""featurize: матрица признаков."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("featurize")
IN = {"train": Path("data/processed/train.csv"), "test": Path("data/processed/test.csv")}
OUT = {"train": Path("data/features/train.csv"), "test": Path("data/features/test.csv")}


def build(frame: pd.DataFrame, target: str, drop: list[str]) -> pd.DataFrame:
    features = frame.drop(columns=[c for c in drop if c in frame.columns], errors="ignore")
    features["rooms_per_person"] = features["AveRooms"] / features["AveOccup"].clip(lower=1)
    ordered = [c for c in features.columns if c != target]
    if target in features.columns:
        ordered.append(target)
    return features[ordered]


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    drop = list(params["data"]["drop"])

    for part, path in IN.items():
        frame = pd.read_csv(path)
        built = build(frame, target, drop)
        built.to_csv(ensure_parent(OUT[part]), index=False)
        log.info("%s: %d строк, признаков %d", part, len(built), built.shape[1] - 1)


if __name__ == "__main__":
    main()
