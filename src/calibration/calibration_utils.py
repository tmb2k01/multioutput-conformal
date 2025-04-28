from typing import Callable, Dict
import numpy as np


def get_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute finite-sample-adjusted 1-alpha quantile of scores.

    Args:
        scores (np.ndarray): Array of conformal scores (higher = more uncertain).
        alpha (float): Miscoverage level.

    Returns:
        float: Quantile value (q-hat).
    """
    n = len(scores)
    if n == 0:
        return np.inf

    val = np.ceil((n + 1) * (1 - alpha)) / n
    if val > 1:
        return np.inf
    try:
        qhat = np.quantile(scores, val, method="inverted_cdf")
    except TypeError:
        # For older numpy versions
        qhat = np.quantile(scores, val, interpolation="lower")
    return qhat


def compute_qhat_scp_global(nonconformity_scores, true_labels, alpha, clusters=None):
    """
    Compute the q-hat value for the Standard Conformal Prediction global calibration method.

    Args:
        nonconformity_scores: The nonconformity scores.
        clusters: The number of clusters (not used in this method).

    Returns:
        The q-hat value.
    """
    assert nonconformity_scores.ndim in (
        2,
        3,
    ), "Nonconformity scores should be 2D or 3D."
    assert (
        true_labels.shape[0] == nonconformity_scores.shape[0]
    ), "Mismatch between true_labels and nonconformity_scores dimensions."

    if len(nonconformity_scores.shape) == 3:
        assert (
            true_labels.shape[1] == nonconformity_scores.shape[1]
        ), "Mismatch between true_labels and nonconformity_scores dimensions."
        true_scores = np.take_along_axis(
            nonconformity_scores, np.expand_dims(true_labels, axis=2), axis=2
        ).squeeze(axis=2)
        return get_conformal_quantile(true_scores.flatten(), alpha)

    elif len(nonconformity_scores.shape) == 2:
        true_scores = np.take_along_axis(
            nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=1
        ).squeeze(axis=1)
        return get_conformal_quantile(true_scores, alpha)


def compute_qhat_scp_task(nonconformity_scores, true_labels, alpha, clusters=None):
    """
    Compute the q-hat value for the Standard Conformal Prediction task calibration method.

    Args:
        nonconformity_scores: The nonconformity scores.
        clusters: The number of clusters.

    Returns:
        The q-hat value.
    """
    assert nonconformity_scores.ndim == 3, "Nonconformity scores should be 3D."
    assert (
        true_labels.shape == nonconformity_scores.shape[:2]
    ), "Shape mismatch between true_labels and nonconformity_scores."

    true_scores = np.take_along_axis(
        nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=2
    ).squeeze(axis=2)
    return np.apply_along_axis(get_conformal_quantile, 1, true_scores, alpha)


def compute_qhat_ccp_class(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    clusters=None,
) -> np.ndarray:
    """
    Compute fully vectorized q-hat values for CCP class calibration.

    Args:
        nonconformity_scores (np.ndarray): Shape (T, B, C) or (B, C).
        true_labels (np.ndarray): Shape (T, B) or (B,).
        alpha (float): Miscoverage level.

    Returns:
        np.ndarray: q-hat values of shape (T, C) or (C,).
    """

    def vectorized_qhat(task_scores, task_labels, num_classes):
        """
        Compute class-wise quantiles for given task scores and labels.

        Parameters:
        -----------
        task_scores : numpy.ndarray
            A 2D array of shape (B, C) where B is the number of samples and C is the number of classes.
            Each entry represents the score for a specific sample and class.
        task_labels : numpy.ndarray
            A 1D array of shape (B,) containing the true class labels for each sample.
        num_classes : int
            The total number of classes.

        Returns:
            numpy.ndarray: np.ndarray: q-hat values of shape (C,).
        """
        # task_scores: (B, C), task_labels: (B,)
        # Create (B, C) mask where mask[b, c] = True if label[b] == c
        mask = task_labels[:, None] == np.arange(num_classes)[None, :]  # (B, C)

        # Set scores to nan where not matching the class
        selected_scores = np.where(mask, task_scores, np.nan)  # (B, C)

        # Now compute quantiles ignoring nan
        n_valid = np.sum(~np.isnan(selected_scores), axis=0)  # (C,)
        val = np.ceil((n_valid + 1) * (1 - alpha)) / n_valid  # (C,)

        # Compute quantiles per class
        qhat_per_class = np.full(num_classes, np.inf)
        for c in range(num_classes):
            if n_valid[c] == 0 or val[c] > 1:
                qhat_per_class[c] = np.inf
            else:
                class_scores = selected_scores[:, c]
                qhat_per_class[c] = np.nanquantile(
                    class_scores, val[c], method="inverted_cdf"
                )

        return qhat_per_class

    assert nonconformity_scores.ndim in (
        2,
        3,
    ), "Expected nonconformity_scores of shape (B, C) or (T, B, C)"

    if nonconformity_scores.ndim == 3:
        T, B, C = nonconformity_scores.shape
        qhats = np.zeros((T, C))
        for t in range(T):
            qhats[t] = vectorized_qhat(nonconformity_scores[t], true_labels[t], C)
        return qhats
    else:
        B, C = nonconformity_scores.shape
        return vectorized_qhat(nonconformity_scores, true_labels, C)


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
    "ccp_global_clusters": None,
    "ccp_joint_class_repr": None,
}
