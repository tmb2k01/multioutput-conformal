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
):
    """
    Perform calibration on the predictions based on the specified calibration type.

    Args:
        predictions: The predictions to be calibrated.
        calibration_type: The type of calibration to be performed.

    Returns:
        The calibrated predictions.
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
