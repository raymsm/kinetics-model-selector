from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ModelFitResult:
    """Standardized return payload for all model fitting functions."""

    model_name: str
    parameters: Dict[str, float]
    predictions: np.ndarray
    residuals: np.ndarray
    r_squared: float
