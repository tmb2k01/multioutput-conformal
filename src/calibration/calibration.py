from typing import Callable, Dict, Union

from calibration.calibration_type_valid import (
    VALID_CALIBRATION_TYPES,
    VALID_NONCONFORMITY_METHODS,
)
from src.calibration.nonconformity_functions import NONCONFORMITY_FN_DIC
from src.models.high_level_model import HighLevelModel
from src.models.low_level_model import LowLevelModel

CALIBRATION_FN_HIGH_DIC: Dict[str, Callable[..., float]] = {
    "scp_task_thresholds": None,
    "scp_global_threshold": None,
    "ccp_class_thresholds": None,
    "ccp_task_clusters": None,
    "ccp_global_clusters": None,
}

CALIBRATION_FN_LOW_DIC: Dict[str, Callable[..., float]] = {
    "scp_global_threshold": None,
#    "ccp_class_thresholds": None,
#    "ccp_global_clusters": None,
#    "ccp_joint_class_repr": None,
}


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
    for nonconformity_name, nonconformity_fn in NONCONFORMITY_FN_DIC().items():
        nonconformity_scores[nonconformity_name] = nonconformity_fn(scores)

    q_hats = {}
    for calibration_type, calibration_fn in (
        CALIBRATION_FN_HIGH_DIC.items()
        if high_level
        else CALIBRATION_FN_LOW_DIC.items()
    ):
        q_hats[calibration_type] = calibration_fn(nonconformity_scores, true_labels, clusters=clusters)


    return q_hats
