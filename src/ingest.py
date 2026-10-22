"""Имитация выгрузки: срез California Housing по широте.

python -m src.ingest                 #версия из params.yaml
python -m src.ingest --version all   #все районы (v2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("ingest")


def main() -> None:
    params = load_params()
    ing = params["ingest"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=ing["version"])
    parser.add_argument("--source", type=Path, default=Path(ing["source"]))
    parser.add_argument("--out", type=Path, default=Path(params["data"]["raw"]))
    args = parser.parse_args()

    frame = pd.read_csv(args.source)
    threshold = float(ing["latitude_threshold"])

    if args.version == "north":
        sliced = frame[frame["Latitude"] >= threshold].copy()
    elif args.version == "all":
        sliced = frame.copy()
    else:
        raise ValueError(f"неизвестная версия: {args.version}")

    if sliced.empty:
        raise ValueError("пустой срез")

    sliced.to_csv(ensure_parent(args.out), index=False)
    log.info("версия %s: %d строк -> %s", args.version, len(sliced), args.out)


if __name__ == "__main__":
    main()
