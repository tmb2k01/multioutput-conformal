from typing import List, Union
import numpy as np


def _get_prediction_set(
    nonconformity_scores: Union[np.ndarray, List[np.ndarray]],
    thresholds: Union[float, np.ndarray, List[np.ndarray]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Internal helper to compute prediction sets where each class is included if its nonconformity
    score is less than or equal to the corresponding threshold.

    Args:
        nonconformity_scores (Union[np.ndarray, List[np.ndarray]]):
            Nonconformity scores for predictions.
            - Shape (C,) for single-task, where C is the number of classes.
            - List of arrays for multi-task, each of shape (C_t,).

        thresholds (Union[float, np.ndarray, List[np.ndarray]]):
            Thresholds to compare scores against.
            - Scalar or array for single-task.
            - List of arrays for multi-task, matching shape of `nonconformity_scores`.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            Indices of classes included in the prediction set.
            - 1D array of indices for single-task.
            - List of 1D arrays for multi-task.
    """
    if isinstance(nonconformity_scores, np.ndarray):
        assert isinstance(
            thresholds, (float, np.ndarray)
        ), "Thresholds must be a float or np.ndarray for single-task input."
        return np.where(nonconformity_scores <= thresholds)[0]

    elif isinstance(nonconformity_scores, list):
        assert isinstance(
            thresholds, list
        ), "Thresholds must be a list if nonconformity_scores is a list."
        assert len(nonconformity_scores) == len(
            thresholds
        ), "Mismatch between number of tasks in scores and thresholds."
        return [
            np.where(task_scores <= thresholds[i])[0]
            for i, task_scores in enumerate(nonconformity_scores)
        ]

    else:
        raise ValueError("Nonconformity scores must be a 1D array or list of arrays.")


def standard_prediction(
    nonconformity_scores: Union[np.ndarray, List[np.ndarray]],
    q_hat: Union[float, np.ndarray, List[np.ndarray]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Computes prediction sets using standard (global or task-wise) conformal prediction thresholds.
    Can be used for:
    - SCP (global threshold for all classes),
    - Per-task SCP (task-specific thresholds),
    - CCP (class-specific thresholds without clusters).

    Args:
        nonconformity_scores (Union[np.ndarray, List[np.ndarray]]):
            Nonconformity scores.
            - Array of shape (C,) for single-task.
            - List of arrays for multi-task.

        q_hat (Union[float, np.ndarray, List[np.ndarray]]):
            Thresholds for inclusion.
            - Scalar or array for single-task.
            - List of arrays for multi-task.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            Indices of predicted classes per task.
    """
    return _get_prediction_set(nonconformity_scores, q_hat)


def clustered_prediction(
    nonconformity_scores: Union[np.ndarray, List[np.ndarray]],
    q_hat: Union[np.ndarray, List[np.ndarray]],
    clusters: Union[np.ndarray, List[np.ndarray]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Computes prediction sets using class-cluster-specific thresholds.
    This is used for Clustered Class-Conditional Prediction (CCP).

    Args:
        nonconformity_scores (Union[np.ndarray, List[np.ndarray]]):
            Nonconformity scores for predictions.
            - Array of shape (C,) for single-task.
            - List of arrays for multi-task.

        q_hat (Union[np.ndarray, List[np.ndarray]]):
            Quantile thresholds per cluster.
            - Array of shape (K,) for single-task.
            - List of arrays of shape (K_t,) for each task t.

        clusters (Union[np.ndarray, List[np.ndarray]]):
            Cluster assignments for each class.
            - Array of shape (C,) for single-task, with cluster index for each class.
            - List of arrays of shape (C_t,) for each task.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            Indices of predicted classes per task using cluster-specific thresholds.
    """
    if isinstance(nonconformity_scores, np.ndarray):
        assert isinstance(
            q_hat, np.ndarray
        ), "Expected q_hat to be np.ndarray for single-task."
        assert isinstance(
            clusters, np.ndarray
        ), "Expected clusters to be np.ndarray for single-task."
        assert (
            nonconformity_scores.shape[0] == clusters.shape[0]
        ), "Mismatch between number of classes and cluster assignments."
        clustered_qhat = np.array([q_hat[c] for c in clusters])

    elif isinstance(nonconformity_scores, list):
        assert isinstance(
            q_hat, list
        ), "Expected q_hat to be a list for multi-task input."
        assert isinstance(
            clusters, list
        ), "Expected clusters to be a list for multi-task input."
        assert (
            len(nonconformity_scores) == len(q_hat) == len(clusters)
        ), "Mismatch in length between scores, thresholds, and cluster assignments."
        clustered_qhat = [
            np.array([q_hat[i][c] for c in clusters[i]])
            for i in range(len(nonconformity_scores))
        ]

    else:
        raise ValueError("Nonconformity scores must be a 1D or 2D array.")

    return _get_prediction_set(nonconformity_scores, clustered_qhat)
