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
