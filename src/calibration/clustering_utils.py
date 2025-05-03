from typing import List, Tuple

import numpy as np


def quantile_embedding(
    samples: np.ndarray, q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
) -> np.ndarray:
    """
    Compute the q-quantiles of the given samples.

    Args:
        samples (np.ndarray): Input array of samples.
        q (list): List of quantiles to compute.

    Returns:
        np.ndarray: The computed quantile values.
    """
    return np.quantile(samples, q)


def embed_all_classes(
    scores_all: np.ndarray,
    labels: np.ndarray,
    q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    return_cts=False,
):
    """
    Compute class-wise quantile embeddings based on prediction scores.

    Args:
        scores_all (np.ndarray): Ground truth prediction scores. Shape (B,) where B is the batch size.
        labels (np.ndarray): True class labels. Shape (B,).
        q (list): List of quantiles to compute.
        return_cts (bool): Whether to return per-class instance counts.

    Returns:
        np.ndarray: Class-wise embeddings of shape (C, len(q)).
        np.ndarray (optional): Count of instances per class. Shape (C,).
    """
    num_classes = np.max(labels) + 1
    quant_dim = len(q)

    embeddings = np.zeros((num_classes, quant_dim))
    cts = np.bincount(labels, minlength=num_classes)

    for cls in range(num_classes):
        cls_scores = scores_all[labels == cls]
        if cls_scores.size > 0:
            embeddings[cls] = quantile_embedding(cls_scores, q)

    return (embeddings, cts) if return_cts else embeddings


def embed_all_tasks(
    scores_all: np.ndarray,
    labels: np.ndarray,
    q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    return_cts=False,
) -> List[np.ndarray]:
    """
    Compute task-wise and class-wise quantile embeddings from prediction scores.

    Args:
        scores_all (np.ndarray): Ground truth prediction scores for each task. Shape (T, B)
                                 where T is the number of tasks and B is the batch size.
        labels (np.ndarray): Ground truth class labels. Shape (T, B).
        q (Tuple[float, ...]): List of quantiles to compute for the embeddings.
        return_cts (bool): Whether to return per-task instance counts. Default is False.

    Returns:
        List[np.ndarray]: Task-wise class embeddings of shape (T, C, len(q)), where T is the number of tasks.
        List[np.ndarray] (optional): Count of instances per class. Shape (T, C).
    """
    T = scores_all.shape[0]
    embeddings = []
    cts = []

    for task in range(T):
        if return_cts:
            task_embeddings, task_cts = embed_all_classes(
                scores_all[task], labels[task], q=q, return_cts=True
            )
            cts.append(task_cts)
        else:
            task_embeddings = embed_all_classes(
                scores_all[task], labels[task], q=q, return_cts=False
            )
        embeddings.append(task_embeddings)

    return (embeddings, cts) if return_cts else embeddings
