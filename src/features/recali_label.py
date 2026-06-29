from __future__ import annotations

import numpy as np
import pandas as pd


def build_recali_label(
    row: pd.Series,
    *,
    enabled: bool,
    source_column: str,
) -> np.ndarray:
    if not enabled:
        return np.empty((0,), dtype=np.float32)
    if source_column not in row.index:
        raise ValueError(
            f"recali label column {source_column!r} is missing from the dataset"
        )

    value = row[source_column]
    if pd.isna(value):
        raise ValueError(
            f"recali label column {source_column!r} contains a missing value"
        )
    if isinstance(value, (bool, np.bool_)):
        encoded = float(value)
    elif isinstance(value, (int, float, np.integer, np.floating)) and float(value) in {
        0.0,
        1.0,
    }:
        encoded = float(value)
    else:
        raise ValueError(
            f"recali label column {source_column!r} must contain booleans or 0/1, "
            f"got {value!r}"
        )

    return np.asarray([encoded], dtype=np.float32)
