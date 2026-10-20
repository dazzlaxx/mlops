"""Имитация выгрузки данных: срез по кредитному лимиту.

v1: только клиенты с LIMIT_BAL < порога (обучающая версия).
v2: все клиенты (--no-limit).

    python -m src.ingest                    # v1 из params.yaml
    python -m src.ingest --no-limit         # v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("ingest")


def main() -> None:
    params = load_params()
    ingest = params["ingest"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-limit", action="store_true", help="не фильтровать по лимиту (v2)")
    parser.add_argument("--source", type=Path, default=Path(ingest["source"]))
    parser.add_argument("--out", type=Path, default=Path(params["data"]["raw"]))
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # UCI исходный файл имеет строку-заголовок с ID; после read_csv первая колонка может быть unnamed
    if "ID" not in df.columns and df.columns[0] == "Unnamed:_0":
        df = df.rename(columns={"Unnamed:_0": "ID"})

    if not args.no_limit and ingest.get("limit_threshold") is not None:
        threshold = ingest["limit_threshold"]
        df = df[df["LIMIT_BAL"] < threshold]
        log.info("фильтр LIMIT_BAL < %s", threshold)

    if df.empty:
        raise ValueError("после фильтрации не осталось строк")

    df.to_csv(ensure_parent(args.out), index=False)
    log.info("выгрузка: %d строк -> %s", len(df), args.out)


if __name__ == "__main__":
    main()
