from typing import Callable, Dict, Union

from calibration.calibration_type_valid import (
    VALID_CALIBRATION_TYPES,
    VALID_NONCONFORMITY_METHODS,
)
from src.calibration.nonconformity_functions import NONCONFORMITY_FN_DIC

CALIBRATION_FN_DIC: Dict[str, Callable[..., float]] = {
    "scp_task_thresholds": None,
    "scp_global_threshold": None,
    "ccp_class_thresholds": None,
    "ccp_task_clusters": None,
    "ccp_global_clusters": None,
    "ccp_joint_class_repr": None,
}


def calibration(
    predictions,
    calibration_type: VALID_CALIBRATION_TYPES,
    unconformity_method: VALID_NONCONFORMITY_METHODS,
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

    nonconformity_fn = NONCONFORMITY_FN_DIC.get(unconformity_method)
    nonconformity_scores = nonconformity_fn(predictions)

    calibration_fn = CALIBRATION_FN_DIC.get(calibration_type)
    if calibration_fn is None:
        raise ValueError(f"Calibration type is not implemented: {calibration_type}")

    q_hat = calibration_fn(nonconformity_scores, clusters=clusters)

    return q_hat
