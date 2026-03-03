"""Input/output helpers for kinetics model selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("time", "concentration")


def read_kinetics_data(path: str | Path) -> pd.DataFrame:
    """Read kinetics input data and validate required schema."""
    data_path = Path(path)
    df = pd.read_csv(data_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(
            f"Input file '{data_path}' is missing required column(s): {missing_columns}. "
            "Expected columns: time, concentration."
        )

    return df
