from typing import Union

from src.calibration.nonconformity_functions import NONCONFORMITY_FN_DIC
from src.calibration.calibration_utils import (
    CALIBRATION_FN_HIGH_DIC,
    CALIBRATION_FN_LOW_DIC,
)


def calibration(
    scores,
    true_labels,
    high_level: bool = True,
    alpha: float = 0.05,
    clusters: Union[None, int] = None,
) -> dict[str, dict]:
    """
    Perform conformal calibration using various nonconformity and calibration methods.

    Args:
        scores (np.ndarray): The model prediction scores. Shape (B, C) or (T, B, C),
                             where B is the batch size, C is the number of classes,
                             and T is the number of tasks (optional).
        true_labels (np.ndarray): True class labels. Shape (B,) or (T, B).
        high_level (bool): If True, use high-level calibration functions.
        alpha (float): Desired miscoverage level (e.g., 0.05 for 95% coverage).
        clusters (int or None): Number of clusters for cluster-based calibration,
                                or None to disable clustering.

    Returns:
        dict[str, dict]: A nested dictionary of q-hat values with structure:
                         {
                             "calibration_type": {
                                 "nonconformity_type": np.ndarray of q-hats
                             }
                         }
                         Each q-hat is of shape (C,) or (T, C) depending on input.
    """

    nonconformity_scores = {}
    for nonconformity_name, nonconformity_fn in NONCONFORMITY_FN_DIC.items():
        nonconformity_scores[nonconformity_name] = nonconformity_fn(scores)

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
                nonconformity_scores[nonconformity_name], true_labels, alpha, clusters=clusters
            )

    return q_hats
