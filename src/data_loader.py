from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetSummary:
    path: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    numeric_stats: dict[str, dict[str, Any]]
    categorical_counts: dict[str, dict[str, int]]


class ParquetDataLoader:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_parquet(self.path)
        return self.df

    def counts(self) -> dict[str, int]:
        df = self._require_df()

        safe_df = df.copy()
        for col in safe_df.columns:
            safe_df[col] = safe_df[col].apply(
                lambda x: tuple(x) if isinstance(x, np.ndarray) else x
            )

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(safe_df.duplicated().sum()),
        }

    def statistics(self, max_categories: int = 10) -> DatasetSummary:
        df = self._require_df()

        numeric_df = df.select_dtypes(include="number")
        categorical_df = df.select_dtypes(exclude="number")

        numeric_stats = {
            column: {
                "mean": float(numeric_df[column].mean()),
                "std": float(numeric_df[column].std()),
                "min": float(numeric_df[column].min()),
                "median": float(numeric_df[column].median()),
                "max": float(numeric_df[column].max()),
            }
            for column in numeric_df.columns
        }

        categorical_counts = {
            column: df[column].astype(str).value_counts(dropna=False).head(max_categories).to_dict()
            for column in categorical_df.columns
        }

        return DatasetSummary(
            path=str(self.path),
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            dtypes={column: str(dtype) for column, dtype in df.dtypes.items()},
            missing_values=df.isna().sum().astype(int).to_dict(),
            numeric_stats=numeric_stats,
            categorical_counts=categorical_counts,
        )

    def preview(self, n: int = 5) -> pd.DataFrame:
        return self._require_df().head(n)

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.df
