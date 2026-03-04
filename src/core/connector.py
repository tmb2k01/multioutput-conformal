from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from core.utils import convert_multitask_preds


class LabelSpaceConnector(ABC):
    """
    Abstract base class for converting between label spaces.
    """

    def __init__(self, n_classes_per_task: np.ndarray) -> None:
        self.n_classes_per_task = np.asarray(n_classes_per_task, dtype=np.int64)

    @property
    def multipliers(self) -> np.ndarray:
        """
        multipliers[t] = product_{k>t} n_classes[k]
        Used for joint-id encoding/decoding.
        """
        n = self.n_classes_per_task
        mult = np.ones_like(n)
        for t in range(len(n) - 2, -1, -1):
            mult[t] = mult[t + 1] * n[t + 1]
        return mult

    @abstractmethod
    def pred_to_calib(self, y: np.ndarray) -> np.ndarray:
        """
        Convert labels to the representation used by the calibrator/model at that level.
        """
        raise NotImplementedError
    
    @abstractmethod
    def gt_to_calib(self, y: np.ndarray) -> np.ndarray:
        """
        Convert ground truth labels to the representation used by the calibrator/model at that level.
        """
        raise NotImplementedError

@dataclass
class HighToHighLabelConnector(LabelSpaceConnector):
    """
    High -> High
    """
    n_classes_per_task: np.ndarray

    def __post_init__(self) -> None:
        super().__init__(self.n_classes_per_task)

    def pred_to_calib(self, y: np.ndarray) -> np.ndarray:
        return convert_multitask_preds(y)

    def gt_to_calib(self, labels: np.ndarray) -> np.ndarray:
        return labels

@dataclass
class LowToLowLabelConnector(LabelSpaceConnector):
    """
    Low -> Low
    """
    n_classes_per_task: np.ndarray

    def __post_init__(self) -> None:
        super().__init__(self.n_classes_per_task)

    def pred_to_calib(self, y: np.ndarray) -> np.ndarray:
        y = np.concatenate(y, axis=0)
        return np.array(y)

    def gt_to_calib(self, y: np.ndarray) -> np.ndarray:
        task_classes = self.n_classes_per_task
        multiplier = np.array(
            [math.prod(task_classes[i + 1 :]) for i in range(len(task_classes))]
        )
        return (y * multiplier[:, None]).sum(axis=0)

@dataclass
class HighToLowLabelConnector(LabelSpaceConnector):
    """
    High -> Low
    """
    n_classes_per_task: np.ndarray

    def __post_init__(self) -> None:
        super().__init__(self.n_classes_per_task)

    def pred_to_calib(self, y: np.ndarray) -> np.ndarray:
        # TODO: Implement
        pass


@dataclass
class LowToHighLabelConnector(LabelSpaceConnector):
    """
    Low -> High
    """
    n_classes_per_task: np.ndarray

    def __post_init__(self) -> None:
        super().__init__(self.n_classes_per_task)

    def pred_to_calib(self, y: np.ndarray) -> np.ndarray:
        # TODO: Implement
        pass