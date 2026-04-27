from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from calibration.calibration_utils import (
    CALIBRATION_FN_HIGH_DIC,
    CALIBRATION_FN_LOW_DIC,
    CLUSTER_FN_DIC,
)
from calibration.conformal_prediction import clustered_prediction, standard_prediction
from calibration.nonconformity_functions import NONCONFORMITY_FN_DIC
from core.thresholds_io import load_thresholds, save_thresholds
from core.utils import expand_path

from .types import CalibrationFn, Level, ThresholdBundle


# -----------------------------
# Friendly errors
# -----------------------------
class CalibratorConfigError(ValueError):
    """Raised when a calibrator is misconfigured (bad keys, missing functions, etc.)."""


class CalibratorDataError(ValueError):
    """Raised when inputs to fit/predict have incompatible shapes or values."""


class CalibratorStateError(RuntimeError):
    """Raised when calling predict/load/save in the wrong state."""


def _available_keys_msg(mapping_name: str, keys: list[str], limit: int = 30) -> str:
    keys_sorted = sorted(keys)
    shown = keys_sorted[:limit]
    suffix = (
        "" if len(keys_sorted) <= limit else f" (and {len(keys_sorted) - limit} more)"
    )
    return f"Available {mapping_name} keys: {shown}{suffix}"


def _get_from_dict[T](d: dict[str, T], key: str, what: str) -> T:
    try:
        return d[key]
    except KeyError as e:
        msg = f"Unknown {what} key '{key}'. {_available_keys_msg(what, list(d.keys()))}"
        raise CalibratorConfigError(msg) from e


def _round_alpha(alpha: float) -> float:
    if not np.isfinite(alpha):
        raise CalibratorConfigError(f"alpha must be a finite float. Got {alpha!r}.")
    if not (0.0 < float(alpha) < 1.0):
        raise CalibratorConfigError(f"alpha must be in (0, 1). Got {alpha!r}.")
    rounded = round(float(alpha), 2)
    if abs(float(alpha) - rounded) > 1e-4:
        # keep as warning, but not an exception
        print(f"Warning: alpha {alpha} rounded to {rounded}")
    return rounded


# -----------------------------
# Calibrators
# -----------------------------
class BaseCalibrator(ABC):
    """
    Abstract calibrator:
      - computes GT nonconformity scores from model outputs + labels
      - runs a calibration function -> thresholds (ThresholdBundle)
      - saves/loads thresholds

    Children implement:
      - level()
      - fit()
      - predict()
    """

    def __init__(
        self,
        *,
        calibration_fn: CalibrationFn,
        calibration_fn_key: str,
        nonconformity_fn_key: str,
        artifacts_dir: str = "./artifacts",
        alpha: float | None = None,
        load_on_init: bool = False,
    ) -> None:
        if calibration_fn is None:
            raise CalibratorConfigError("calibration_fn must be provided.")
        if not calibration_fn_key:
            raise CalibratorConfigError("calibration_key must be a non-empty string.")
        if not nonconformity_fn_key:
            raise CalibratorConfigError("nonconformity_key must be a non-empty string.")
        if load_on_init and alpha is None:
            raise CalibratorConfigError(
                "alpha must be provided when load_on_init=True."
            )
        self.calibrationFn = calibration_fn
        self.calibration_fn_key = calibration_fn_key

        self.nonconformity_fn_key = nonconformity_fn_key
        self.nonconformityFn = _get_from_dict(
            NONCONFORMITY_FN_DIC, nonconformity_fn_key, what="nonconformity"
        )

        self.artifacts_dir = expand_path(artifacts_dir)
        self.thresholds_root = self.artifacts_dir / "thresholds"

        self.thresholds: ThresholdBundle | None = None

        if load_on_init:
            alpha = _round_alpha(alpha)
            self.load_thresholds(alpha)

    def _thr_path(self, alpha: float) -> str:
        base = (
            f"{self.level}/"
            f"{self.nonconformity_fn_key}/"
            f"{self.calibration_fn_key}/"
            f"alpha_{alpha:.2f}.json"
        )
        return f"{self.thresholds_root}/{base}" if self.thresholds_root else base

    @property
    @abstractmethod
    def level(self) -> Level:
        raise NotImplementedError

    def fit(
        self, model_outputs: Any, labels: np.ndarray, alpha: float, **kwargs: dict
    ) -> ThresholdBundle:
        alpha = _round_alpha(alpha)

        if self.calibrationFn in CLUSTER_FN_DIC.values():
            n_clusters = kwargs.get("n_clusters", "auto")

        gt_scores = self.compute_gt_nonconformity(model_outputs, labels)

        q = self.calibrationFn(
            gt_scores,
            labels,
            alpha,
            n_clusters=(
                n_clusters if self.calibrationFn in CLUSTER_FN_DIC.values() else None
            ),
        )

        payload = {"q_hats": q}

        bundle = ThresholdBundle(
            level=self.level,
            nonconformity_key=self.nonconformity_fn_key,
            calibration_key=self.calibration_fn_key,
            alpha=alpha,
            payload=payload,
        )
        self.thresholds = bundle
        self._save_thresholds(bundle)
        return bundle

    def predict(self, batch_outputs: Any) -> list[np.ndarray] | list[list[np.ndarray]]:
        if self.thresholds is None:
            raise CalibratorStateError(
                "Thresholds are not available. Call fit(...) or load_thresholds(...)."
            )

        scores = self.get_prediction_scores(batch_outputs)
        payload = self.thresholds.payload
        q_hats = payload["q_hats"]

        if not isinstance(q_hats, dict):
            if np.isscalar(q_hats):
                q_hats = float(q_hats)
            elif not np.isscalar(q_hats[0]):
                q_hats = (
                    float(q_hats)
                    if np.isscalar(q_hats)
                    else [np.asarray(t_qhats).reshape(-1) for t_qhats in payload["q_hats"]]
                )
            return standard_prediction(scores, q_hats)  # broadcast (B,) -> (B,C)

        # Clustered thresholds (clustered CCP)
        return clustered_prediction(scores, q_hats)

    @abstractmethod
    def compute_gt_nonconformity(
        self, outputs: list[np.ndarray] | np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def get_prediction_scores(
        self, batch_outputs: Any
    ) -> list[np.ndarray] | np.ndarray:
        raise NotImplementedError

    def load_thresholds(self, alpha: float) -> ThresholdBundle:
        alpha = _round_alpha(alpha)
        path = self._thr_path(alpha)
        try:
            self.thresholds = load_thresholds(path)
        except FileNotFoundError as e:
            raise CalibratorStateError(
                f"Threshold file not found at '{path}'. "
                f"Run fit(...) first or verify thresholds_root/keys/alpha."
            ) from e
        except Exception as e:
            raise CalibratorStateError(
                f"Failed to load thresholds from '{path}': {e}"
            ) from e
        return self.thresholds

    def _save_thresholds(self, bundle: ThresholdBundle) -> None:
        path = self._thr_path(bundle.alpha)
        try:
            save_thresholds(bundle, path)
        except Exception as e:
            raise CalibratorStateError(
                f"Failed to save thresholds to '{path}': {e}"
            ) from e


@dataclass
class HighLevelCalibrator(BaseCalibrator):
    calibration_fn_key: str
    nonconformity_fn_key: str
    load_on_init: bool = False
    alpha: float = 0.05
    artifacts_dir: str = "./artifacts"

    def __post_init__(self) -> None:
        calibrationFn = _get_from_dict(
            CALIBRATION_FN_HIGH_DIC,
            self.calibration_fn_key,
            what="high-level calibration",
        )
        super().__init__(
            calibration_fn=calibrationFn,
            calibration_fn_key=self.calibration_fn_key,
            nonconformity_fn_key=self.nonconformity_fn_key,
            alpha=self.alpha,
            load_on_init=self.load_on_init,
            artifacts_dir=self.artifacts_dir,
        )

    level = "high"

    def get_prediction_scores(self, batch_outputs):
        scores = self.nonconformityFn(batch_outputs)
        return [np.asarray(task_scores) for task_scores in scores]

    def compute_gt_nonconformity(
        self, outputs: list[np.ndarray], labels: np.ndarray
    ) -> list[np.ndarray]:
        taskwise = []
        for task_score, task_label in zip(outputs, labels, strict=False):
            full_scores = self.nonconformityFn(task_score)
            gt_scores = full_scores[np.arange(task_score.shape[0]), task_label]  # (B,)
            taskwise.append(gt_scores)
        return np.stack(taskwise)


@dataclass
class LowLevelCalibrator(BaseCalibrator):
    calibration_fn_key: str
    nonconformity_fn_key: str
    load_on_init: bool = False
    alpha: float = 0.05
    artifacts_dir: str = "./artifacts"

    def __post_init__(self) -> None:
        calibration_fn = _get_from_dict(
            CALIBRATION_FN_LOW_DIC, self.calibration_fn_key, what="low-level calibration"
        )
        super().__init__(
            calibration_fn=calibration_fn,
            calibration_fn_key=self.calibration_fn_key,
            nonconformity_fn_key=self.nonconformity_fn_key,
            alpha=self.alpha,
            load_on_init=self.load_on_init,
            artifacts_dir=self.artifacts_dir,
        )

    level = "low"

    def get_prediction_scores(self, batch_outputs):
        return np.asarray(self.nonconformityFn(batch_outputs))

    def compute_gt_nonconformity(
        self, outputs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        full_scores = self.nonconformityFn(outputs)  # (B, C)
        return full_scores[np.arange(outputs.shape[0]), labels]  # (B,)
