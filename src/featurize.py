"""featurize: one-hot кодирование категориальных признаков."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("featurize")

IN = {"train": Path("data/processed/train.csv"), "test": Path("data/processed/test.csv")}
OUT = {"train": Path("data/features/train.csv"), "test": Path("data/features/test.csv")}


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    drop_cols = params["data"].get("drop", [])
    one_hot = params["featurize"].get("one_hot", [])

    train_raw = pd.read_csv(IN["train"])
    categories = {
        c: sorted(train_raw[c].dropna().unique())
        for c in one_hot
        if c in train_raw.columns
    }

    for part, path in IN.items():
        frame = pd.read_csv(path)
        frame = frame.drop(
            columns=[c for c in drop_cols if c in frame.columns], errors="ignore"
        )

        for col, cats in categories.items():
            if col in frame.columns:
                frame[col] = pd.Categorical(frame[col], categories=cats)

        frame = pd.get_dummies(frame, columns=list(categories), drop_first=True)
        frame.to_csv(ensure_parent(OUT[part]), index=False)
        log.info("%s: %d строк, %d колонок", part, len(frame), frame.shape[1])


if __name__ == "__main__":
    main()
