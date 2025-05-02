from typing import Dict, List, Union

import numpy as np

from src.calibration.calibration_utils import (
    CALIBRATION_FN_HIGH_DIC,
    CALIBRATION_FN_LOW_DIC,
)
from src.calibration.nonconformity_functions import NONCONFORMITY_FN_DIC


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

    def compute_nonconformity(
        scores: Union[List[np.ndarray], np.ndarray],
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Compute nonconformity scores for the given scores.
        Args:
            scores (Union[List[np.ndarray], np.ndarray]): The model prediction scores.
                - Shape (B, C) or (T, B, C) for multiple tasks.
                - Shape (B, C) for a single task.
        Returns:
            Dict[str, Union[np.ndarray, List[np.ndarray]]]: A dictionary of nonconformity scores.
                Each key corresponds to a nonconformity function and the value is the computed score.
                - Shape (B, C) for a single task.
                - Shape (T, B, C) for multiple tasks.
        """
        return {
            nonconformity_name: (
                [nonconformity_fn(task_scores) for task_scores in scores]
                if isinstance(scores, list)
                else nonconformity_fn(scores)
            )
            for nonconformity_name, nonconformity_fn in NONCONFORMITY_FN_DIC.items()
        }

    nonconformity_scores = compute_nonconformity(scores)

    q_hats = {}
    for calibration_type, calibration_fn in (
        CALIBRATION_FN_HIGH_DIC.items()
        if high_level
        else CALIBRATION_FN_LOW_DIC.items()
    ):
        for nonconformity_name in NONCONFORMITY_FN_DIC.keys():
            if calibration_type not in q_hats:
                q_hats[calibration_type] = {}

            q_hats[calibration_type][nonconformity_name] = calibration_fn(
                nonconformity_scores[nonconformity_name], true_labels, alpha, **kwargs
            )

    return q_hats
