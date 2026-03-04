from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Level = Literal["high", "low"]
NonconformityKey = Literal["hinge", "margin", "pip"]
CalibrationFn = Callable[..., np.ndarray]
NonconformityFn = Callable[..., np.ndarray]

def _jsonify(x: Any) -> Any:
    """Recursively convert common non-JSON types (e.g., numpy) into JSON-serializable Python types."""
    # numpy scalars (np.float32, np.int64, etc.)
    if isinstance(x, np.generic):
        return x.item()

    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()

    # dict-like
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]

    # sets -> list
    if isinstance(x, set):
        return [_jsonify(v) for v in x]

    # pass through for JSON-native types (str/int/float/bool/None) and anything already serializable
    return x

@dataclass(frozen=True)
class ThresholdBundle:
    """
    Generic container for calibration results.
    - level: 'high' or 'low'
    - nonconformity_key: key used from NONCONFORMITY_FN_DIC
    - calibration_key: key used from CALIBRATION_FN_*_DIC
    - alpha: calibration miscoverage level
    - payload: arbitrary json-serializable structure (what your current code saves)
    """
    level: Level
    nonconformity_key: str
    calibration_key: str
    alpha: float
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "nonconformity_key": self.nonconformity_key,
            "calibration_key": self.calibration_key,
            "alpha": self.alpha,
            "payload": _jsonify(self.payload),
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> ThresholdBundle:
        return ThresholdBundle(
            level=d["level"],
            nonconformity_key=d["nonconformity_key"],
            calibration_key=d["calibration_key"],
            alpha=float(d["alpha"]),
            payload=dict(d["payload"]),
        )

