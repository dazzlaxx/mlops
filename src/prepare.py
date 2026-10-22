"""prepare: проверка схемы и разбиение на train/test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schema import validate_raw
from src.utils import ensure_parent, get_logger, load_params

log = get_logger("prepare")
TRAIN_OUT = Path("data/processed/train.csv")
TEST_OUT = Path("data/processed/test.csv")


def main() -> None:
    params = load_params()
    dp = params["data"]
    target = dp["target"]

    raw = pd.read_csv(dp["raw"])
    validated = validate_raw(raw)
    log.info("схема пройдена: %d строк, %d колонок", len(validated), validated.shape[1])

    if validated[target].isna().any():
        raise ValueError(f"пропуски в {target}")

    if bool(dp["split_by_time"]):
        raise ValueError("для этого набора split_by_time должен быть false")

    seed = int(params["train"]["seed"])
    test_size = float(dp["test_size"])
    shuffled = validated.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    edge = int(len(shuffled) * (1 - test_size))
    train, test = shuffled.iloc[:edge], shuffled.iloc[edge:]

    train.to_csv(ensure_parent(TRAIN_OUT), index=False)
    test.to_csv(ensure_parent(TEST_OUT), index=False)
    log.info("train %d, test %d", len(train), len(test))


if __name__ == "__main__":
    main()
