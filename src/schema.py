"""Схема исходных данных California Housing."""

from __future__ import annotations

import pandas as pd
import pandera as pa

SCHEMA = pa.DataFrameSchema(
    {
        "MedInc": pa.Column(float),
        "HouseAge": pa.Column(float),
        "AveRooms": pa.Column(float),
        "AveBedrms": pa.Column(float),
        "Population": pa.Column(float),
        "AveOccup": pa.Column(float),
        "Latitude": pa.Column(float),
        "Longitude": pa.Column(float),
        "MedHouseVal": pa.Column(float),
    },
    strict=False,
)


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    return SCHEMA.validate(frame)
