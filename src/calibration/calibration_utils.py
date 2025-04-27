from typing import Callable, Dict
import numpy as np


def get_conformal_quantile(scores, alpha):
    """
    Compute finite-sample-adjusted 1-alpha quantile of scores

    Inputs:
        - scores: num_instances-length array of conformal scores for true class. A higher score
            indicates more uncertainty
        - alpha: float between 0 and 1 specifying coverage level
    """
    n = len(scores)

    if n == 0:
        print(f"Using default q_hat (np.inf) because n={n}")
        return np.inf

    val = np.ceil((n + 1) * (1 - alpha)) / n
    if val > 1:
        print(f"Using default q_hat (np.inf) because n={n}")
        qhat = np.inf
    else:
        qhat = np.quantile(scores, val, method="inverted_cdf")

    return qhat


def compute_qhat_scp_global(nonconformity_scores, true_labels, alpha, clusters=None):
    """
    Compute the q-hat value for the SCP global calibration method.

    Args:
        nonconformity_scores: The nonconformity scores.
        clusters: The number of clusters (not used in this method).

    Returns:
        The q-hat value.
    """
    assert (
        len(nonconformity_scores.shape) == 2 or len(nonconformity_scores.shape) == 3
    ), "Nonconformity Scores should be 2D or 3D"

    if len(nonconformity_scores.shape) == 3:
        true_scores = np.squeeze(
            np.take_along_axis(
                nonconformity_scores, np.expand_dims(true_labels, axis=2), axis=2
            )
        )
        return get_conformal_quantile(true_scores.flatten(), alpha)

    elif len(nonconformity_scores.shape) == 2:
        true_scores = np.squeeze(
            np.take_along_axis(
                nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=1
            )
        )
        return get_conformal_quantile(true_scores, alpha)


def compute_qhat_scp_task(nonconformity_scores, true_labels, alpha, clusters=None):
    """
    Compute the q-hat value for the SCP task calibration method.

    Args:
        nonconformity_scores: The nonconformity scores.
        clusters: The number of clusters.

    Returns:
        The q-hat value.
    """
    assert (
        len(nonconformity_scores.shape) == 2 or len(nonconformity_scores.shape) == 3
    ), "Nonconformity Scores should be 3D"

    true_scores = np.take_along_axis(
        nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=2
    ).squeeze(axis=2)
    return np.apply_along_axis(get_conformal_quantile, 1, true_scores, alpha)


def compute_qhat_ccp_class(nonconformity_scores, true_labels, alpha, clusters=None):
    """
    Compute the q-hat value for the CCP class calibration
    method.
    Args:
        nonconformity_scores: The nonconformity scores.
        clusters: The number of clusters.
    Returns:
        The q-hat value.
    """
    assert (
        len(nonconformity_scores.shape) == 2 or len(nonconformity_scores.shape) == 3
    ), "Nonconformity Scores should be 3D"

    def compute_qhat_for_task(task_scores, task_labels):
        unique_labels = np.unique(task_labels)
        qhat_per_class = np.zeros(unique_labels.shape)
        for idx, label in enumerate(unique_labels):
            label_scores = task_scores[task_labels == label]
            qhat_per_class[idx] = get_conformal_quantile(label_scores.flatten(), alpha)
        return qhat_per_class

    if len(nonconformity_scores.shape) == 3:
        num_tasks = nonconformity_scores.shape[0]
        qhat_per_task = np.zeros((num_tasks, nonconformity_scores.shape[2]))
        for task_idx in range(num_tasks):
            task_scores = nonconformity_scores[task_idx]
            task_labels = true_labels[task_idx]
            qhat_per_task[task_idx] = compute_qhat_for_task(task_scores, task_labels)
        return qhat_per_task

    elif len(nonconformity_scores.shape) == 2:
        return compute_qhat_for_task(nonconformity_scores, true_labels)


CALIBRATION_FN_HIGH_DIC: Dict[str, Callable[..., float]] = {
    "scp_global_threshold": compute_qhat_scp_global,
    "scp_task_thresholds": compute_qhat_scp_task,
    "ccp_class_thresholds": compute_qhat_ccp_class,
    "ccp_task_cluster_thresholds": None,
    "ccp_global_cluster_thresholds": None,
}

CALIBRATION_FN_LOW_DIC: Dict[str, Callable[..., float]] = {
    "scp_global_threshold": compute_qhat_scp_global,
    "ccp_class_thresholds": compute_qhat_ccp_class,
    #    "ccp_global_clusters": None,
    #    "ccp_joint_class_repr": None,
}
