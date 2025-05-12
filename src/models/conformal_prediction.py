from typing import List, Union

import numpy as np
import torch


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
            - Shape (B, C) for single-task, where B is the batch size and C is the number of classes.
            - List of arrays for multi-task, each of shape (B, C_t).

        thresholds (Union[float, np.ndarray, List[np.ndarray]]):
            Thresholds to compare scores against.
            - Scalar or array for single-task.
            - List of arrays for multi-task, matching shape of `nonconformity_scores`.

    Returns:
        List[List[np.ndarray]]:
            Indices of classes included in the prediction set.
            - For single-task: a list containing one list of 1D arrays (one per sample).
            - For multi-task: a list of lists, each containing 1D arrays of indices per sample for each task.
    """

    def compute_task(task_scores, task_thresholds):
        task_scores = np.asarray(task_scores)
        if np.isscalar(task_thresholds):
            task_thresholds = np.full(task_scores.shape, task_thresholds)
        else:
            task_thresholds = np.broadcast_to(task_thresholds, task_scores.shape)
        return [
            np.where(row <= thresh)[0]
            for row, thresh in zip(task_scores, task_thresholds)
        ]

    if isinstance(nonconformity_scores, np.ndarray):
        return [compute_task(nonconformity_scores, thresholds)]

    elif isinstance(nonconformity_scores, list):
        if np.isscalar(thresholds) or isinstance(thresholds, np.ndarray):
            thresholds = [thresholds] * len(nonconformity_scores)
        return [
            compute_task(scores, thresh)
            for scores, thresh in zip(nonconformity_scores, thresholds)
        ]

    else:
        raise ValueError("nonconformity_scores must be an ndarray or list of ndarrays.")


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
            - Shape (B, C) for single-task, where B is the batch size and C is the number of classes.
            - List of arrays for multi-task, each of shape (B, C_t).

        q_hat (Union[float, np.ndarray, List[np.ndarray]]):
            Thresholds for inclusion.
            - Scalar or array for single-task.
            - List of arrays for multi-task.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            Indices of predicted classes per task.
    """
    return _get_prediction_set(nonconformity_scores, q_hat)


def _extract_qhat_and_clusters(data: dict):
    """
    Extracts q_hat and cluster mappings from the provided configuration.
    This is used for Clustered Class-Conditional Prediction (CCP).
    Args:
        config (dict): Configuration dictionary containing q_hat and cluster mappings.
    Returns:
        Tuple[np.ndarray, List[np.ndarray]]:
            - q_hat: List of class-specific thresholds for each task.
            - clusters: List of cluster assignments for each task.
    """

    if "cluster_qhats" in data:
        q_hat_shared = data["cluster_qhats"]
        q_hat = []
        clusters = []
        for task_id in sorted(data["class_to_cluster_mapping"].keys()):
            clusters.append(data["class_to_cluster_mapping"][task_id])
            q_hat.append(q_hat_shared)
        return q_hat, clusters
    else:
        q_hat = []
        clusters = []
        for task_id in sorted(data.keys()):
            q_hat.append(data[task_id]["qhats"])
            clusters.append(data[task_id]["mapping"])
        return q_hat, clusters


def clustered_prediction(
    nonconformity_scores: Union[np.ndarray, List[np.ndarray]],
    q_hat_data: Union[np.ndarray, List[np.ndarray]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Computes prediction sets using class-cluster-specific thresholds.
    This is used for Clustered Class-Conditional Prediction (CCP).

    Args:
        nonconformity_scores (Union[np.ndarray, List[np.ndarray]]):
            Nonconformity scores for predictions.
            - Shape (B, C) for single-task, where B is the batch size and C is the number of classes.
            - List of arrays for multi-task, each of shape (B, C_t).

        q_hat_data (Union[np.ndarray, List[np.ndarray]]):
            Object which contains class to cluster mappings and quantile thresholds per cluster.

    Returns:
        Union[np.ndarray, List[np.ndarray]]:
            Indices of predicted classes per task using cluster-specific thresholds.
    """
    if isinstance(nonconformity_scores, np.ndarray):
        # q_hat = TODO
        # clusters = TODO
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
        q_hat, clusters = _extract_qhat_and_clusters(q_hat_data)
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
