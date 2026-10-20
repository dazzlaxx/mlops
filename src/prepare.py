"""prepare: проверка и стратифицированное разбиение."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_parent, get_logger, load_params, set_seed

log = get_logger("prepare")

TRAIN_OUT = Path("data/processed/train.csv")
TEST_OUT = Path("data/processed/test.csv")


def main() -> None:
    params = load_params()
    data = params["data"]
    target = data["target"]
    seed = int(data["seed"])

    frame = pd.read_csv(data["raw"])
    log.info("исходные данные: %d строк, %d колонок", len(frame), frame.shape[1])

    if target not in frame.columns:
        raise ValueError(f"целевая колонка {target} не найдена")

    train, test = train_test_split(
        frame,
        test_size=float(data["test_size"]),
        random_state=seed,
        stratify=frame[target] if data.get("stratify") else None,
    )

    train.to_csv(ensure_parent(TRAIN_OUT), index=False)
    test.to_csv(ensure_parent(TEST_OUT), index=False)
    log.info("обучение: %d строк, тест: %d строк", len(train), len(test))


if __name__ == "__main__":
    main()
