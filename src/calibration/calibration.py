from typing import Dict, List, Union

import numpy as np

from src.calibration.calibration_utils import (
    CALIBRATION_FN_HIGH_DIC,
    CALIBRATION_FN_LOW_DIC,
)
from src.calibration.nonconformity_functions import NONCONFORMITY_FN_DIC


def compute_gt_nonconformity(
    scores: Union[List[np.ndarray], np.ndarray],
    labels: np.ndarray,
) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
    """
    Compute nonconformity scores for the given scores, then return only the ground truth scores.
    Args:
        scores (Union[List[np.ndarray], np.ndarray]): The model prediction scores.
            - Shape (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.
        labels (np.ndarray): True class labels.
            - Shape (T, B) for multiple tasks.
            - Shape (B,) for a single task.
    Returns:
        Dict[str, np.ndarray]: A dictionary of nonconformity scores.
            Each key corresponds to a nonconformity function and the value is the computed score.
            - Shape (B,) for a single task.
            - Shape (T, B) for multiple tasks.
    """
    nonconformity_scores = {}

    for name, fn in NONCONFORMITY_FN_DIC.items():
        if isinstance(scores, list):
            # Multi-task: iterate per task
            taskwise = []
            for task_score, task_label in zip(scores, labels):
                full_scores = fn(task_score)  # (B, C_t)
                gt_scores = full_scores[
                    np.arange(task_score.shape[0]), task_label
                ]  # (B,)
                taskwise.append(gt_scores)
            nonconformity_scores[name] = np.stack(taskwise)  # shape (T, B)
        else:
            # Single-task
            full_scores = fn(scores)  # (B, C)
            gt_scores = full_scores[np.arange(scores.shape[0]), labels]  # (B,)
            nonconformity_scores[name] = gt_scores  # shape (B,)

    return nonconformity_scores


def calibration(
    scores: Union[List[np.ndarray], np.ndarray],
    true_labels: np.ndarray,
    high_level: bool = True,
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Perform conformal calibration using various nonconformity and calibration methods.

    Args:
        scores (Union[List[np.ndarray], np.ndarray]): The model prediction scores.
            - Shape (B, C) or (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.
        true_labels (np.ndarray): True class labels. Shape (B,) or (T, B).
        high_level (bool): If True, use high-level calibration functions.
        alpha (float): Desired miscoverage level (e.g., 0.05 for 95% coverage).
        **kwargs (dict): Additional keyword arguments to be passed to individual calibration functions,
                         e.g., clusters, cluster_method, etc.

    Returns:
        dict[str, dict]: A nested dictionary of q-hat values with structure:
                         {
                             "calibration_type": {
                                 "nonconformity_type": np.ndarray of q-hats
                             }
                         }
                         Each q-hat is of shape (C,) or (T, C) depending on input.
    """
    nonconformity_scores = compute_gt_nonconformity(scores, true_labels)

    q_hats = {}
    for calibration_type, calibration_fn in (
        CALIBRATION_FN_HIGH_DIC.items()
        if high_level
        else CALIBRATION_FN_LOW_DIC.items()
    ):
        for nonconformity_name in NONCONFORMITY_FN_DIC.keys():
            if nonconformity_name not in q_hats:
                q_hats[nonconformity_name] = {}

            q_hats[nonconformity_name][calibration_type] = calibration_fn(
                nonconformity_scores[nonconformity_name], true_labels, alpha, **kwargs
            )

    return q_hats
