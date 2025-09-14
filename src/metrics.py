from typing import List

import numpy as np


def compute_informativeness(predictions: List[np.ndarray]) -> float:
    """
    Computes informativeness as the proportion of singleton prediction sets.

    Args:
        predictions (List[np.ndarray]): A list of prediction arrays per sample for a single task.

    Returns:
        float: Fraction of predictions that contain exactly one label.
    """
    return sum(1 for pred in predictions if pred.size == 1) / len(predictions)


def compute_taskwise_informativeness(predictions: List[List[np.ndarray]]) -> np.ndarray:
    """
    Computes informativeness for each task individually.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of prediction arrays per sample.

    Returns:
        np.ndarray: An array containing informativeness for each task.
    """
    return np.array([compute_informativeness(task_preds) for task_preds in predictions])


def compute_overall_informativeness(predictions: List[List[np.ndarray]]) -> float:
    """
    Computes the average informativeness across all tasks.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of prediction arrays per sample.

    Returns:
        float: Mean informativeness across all tasks.
    """
    return np.mean(compute_taskwise_informativeness(predictions))


def compute_efficiency(predictions: List[np.ndarray]) -> float:
    """
    Computes efficiency as the average size of the prediction sets.

    Args:
        predictions (List[np.ndarray]): A list of prediction arrays per sample for a single task.

    Returns:
        float: Average number of predicted labels per sample.
    """
    return sum(len(pred) for pred in predictions) / len(predictions)


def compute_taskwise_efficiency(predictions: List[List[np.ndarray]]) -> np.ndarray:
    """
    Computes efficiency for each task individually.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of prediction arrays per sample.

    Returns:
        np.ndarray: An array containing efficiency for each task.
    """
    return np.array([compute_efficiency(task_preds) for task_preds in predictions])


def compute_overall_efficiency(predictions: List[List[np.ndarray]]) -> float:
    """
    Computes the average efficiency across all tasks.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of prediction arrays per sample.

    Returns:
        float: Mean efficiency across all tasks.
    """
    return np.mean(compute_taskwise_efficiency(predictions))


def compute_weighted_efficiency(
    predictions: List[List[np.ndarray]],
    task_weights: np.ndarray,
    normalize_weights: bool = True,
) -> float:
    """
    Computes weighted efficiency across tasks based on provided weights.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of prediction arrays per sample.
        task_weights (np.ndarray): Weights for each task. Should have same length as number of tasks.
        normalize_weights (bool): If True, weights will be normalized to sum to 1.

    Returns:
        float: Weighted average efficiency across tasks.
    """
    taskwise_eff = compute_taskwise_efficiency(predictions)

    if normalize_weights:
        task_weights = task_weights / np.sum(task_weights)

    return np.sum(taskwise_eff * task_weights)


def compute_weighted_informativeness(
    predictions: List[List[np.ndarray]],
    task_weights: np.ndarray,
    normalize_weights: bool = True,
) -> float:
    """
    Computes weighted informativeness across tasks based on provided weights.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of prediction arrays per sample.
        task_weights (np.ndarray): Weights for each task. Should have same length as number of tasks.
        normalize_weights (bool): If True, weights will be normalized to sum to 1.

    Returns:
        float: Weighted average informativeness across tasks.
    """
    taskwise_info = compute_taskwise_informativeness(predictions)

    if normalize_weights:
        task_weights = task_weights / np.sum(task_weights)

    return np.sum(taskwise_info * task_weights)


def compute_class_based_weights(task_num_classes: List[int]) -> np.ndarray:
    """
    Computes task weights based on the number of classes in each task.
    Tasks with fewer classes get lower weights.

    Args:
        task_num_classes (List[int]): List of number of classes for each task.

    Returns:
        np.ndarray: Weights for each task, normalized to sum to 1.
    """
    weights = np.array(task_num_classes)
    return weights / np.sum(weights)


def compute_classwise_coverage(
    predictions: List[np.ndarray], labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Computes empirical class-conditional coverage for a single task.

    Args:
        predictions (List[np.ndarray]): Prediction sets for each sample.
        labels (np.ndarray): True labels for each sample.
        num_classes (int): Number of classes in the task.

    Returns:
        np.ndarray: Classwise coverage rates (ĉ_y for each class y).
    """
    coverages = []
    for y in range(num_classes):
        idxs = np.where(labels == y)[0]
        if len(idxs) == 0:
            continue  # skip classes not present in the validation set
        cov_y = np.mean([labels[i] in predictions[i] for i in idxs])
        coverages.append(cov_y)
    return np.array(coverages)


def compute_overall_covgap(
    predictions: List[List[np.ndarray]],
    labels: List[np.ndarray],
    task_num_classes: List[int],
    alpha: float,
) -> float:
    """
    Computes average class coverage gap (CovGap) across tasks.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing prediction sets per sample.
        labels (List[np.ndarray]): List of true label arrays for each task.
        task_num_classes (List[int]): Number of classes for each task.
        alpha (float): Significance level.

    Returns:
        float: CovGap value.
    """
    all_coverages = []
    for preds, lbls, C in zip(predictions, labels, task_num_classes):
        class_cov = compute_classwise_coverage(preds, lbls, C)
        all_coverages.extend(class_cov)

    all_coverages = np.array(all_coverages)
    return 100 * np.mean(np.abs(all_coverages - (1 - alpha)))


def compute_covgap(predictions, labels, num_classes, alpha):
    """
    Compute average class coverage gap (CovGap) for a single task.
    """
    class_cov = compute_classwise_coverage(predictions, labels, num_classes)
    return 100 * np.mean(np.abs(class_cov - (1 - alpha)))
