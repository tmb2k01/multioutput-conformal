from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from core.utils import convert_multitask_preds, to_cpu


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
    def pred_to_calib(self, preds: np.ndarray) -> np.ndarray:
        """
        Convert labels to the representation used by the calibrator/model at that level.
        """
        raise NotImplementedError

    @abstractmethod
    def gt_to_calib(self, y: np.ndarray) -> np.ndarray:
        """
        Convert ground truth labels to the representation used by the 
        calibrator/model at that level.
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

    def pred_to_calib(self, preds: list[torch.Tensor]) -> np.ndarray:
        return convert_multitask_preds(preds)

    def gt_to_calib(self, y: np.ndarray) -> np.ndarray:
        return y


@dataclass
class LowToLowLabelConnector(LabelSpaceConnector):
    """
    Low -> Low
    """

    n_classes_per_task: np.ndarray

    def __post_init__(self) -> None:
        super().__init__(self.n_classes_per_task)

    def pred_to_calib(self, preds: np.ndarray) -> np.ndarray:
        preds = np.concatenate(to_cpu(preds), axis=0)
        return np.array(preds)

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

    def pred_to_calib(self, preds: list[torch.Tensor]) -> np.ndarray:
        preds = convert_multitask_preds(preds)
        result = preds[0]
        for arr in preds[1:]:
            result = np.einsum("ni,nj->nij", result, arr).reshape(result.shape[0], -1)
        return result

    def gt_to_calib(self, y: np.ndarray) -> np.ndarray:
        task_classes = self.n_classes_per_task
        multiplier = np.array(
            [math.prod(task_classes[i + 1 :]) for i in range(len(task_classes))]
        )
        return (y * multiplier[:, None]).sum(axis=0)


@dataclass
class LowToHighLabelConnector(LabelSpaceConnector):
    """
    Low -> High
    """

    n_classes_per_task: np.ndarray

    def __post_init__(self) -> None:
        super().__init__(self.n_classes_per_task)

    def pred_to_calib(self, preds: np.ndarray) -> np.ndarray:
        preds = np.concatenate(to_cpu(preds), axis=0)
        task_classes = self.n_classes_per_task
        n_tasks = len(task_classes)
        total_dim = int(np.prod(task_classes))

        if preds.shape[1] != total_dim:
            raise ValueError(
                f"Expected preds.shape[1] == {total_dim}, got {preds.shape[1]}"
            )
        joint = preds.reshape(preds.shape[0], *task_classes)

        result = []
        for task_idx in range(n_tasks):
            axes_to_sum = tuple(
                ax for ax in range(1, n_tasks + 1) if ax != task_idx + 1
            )
            marginal = joint.sum(axis=axes_to_sum)
            result.append(marginal)

        return result

    def gt_to_calib(self, y: np.ndarray) -> np.ndarray:
        return y
