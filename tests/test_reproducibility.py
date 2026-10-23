import json
from pathlib import Path

import pytest

METRICS = Path("reports/metrics.json")
TOL = 1e-4
EXPECTED = {"r2": 0.83, "mae": 0.318, "rmse": 0.478, "n_test": 4128}


def test_metrics_file_exists():
    assert METRICS.exists(), "reports/metrics.json нет — сначала dvc repro"


def test_metrics_in_range():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert 0.0 <= metrics["r2"] <= 1.0
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["n_test"] > 0


def test_metrics_stable():
    """Метрики совпадают с зафиксированными с точностью до допуска."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    for key, value in EXPECTED.items():
        assert metrics[key] == pytest.approx(value, abs=TOL), key
