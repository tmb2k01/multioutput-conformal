import numpy as np


def quantile_embedding(samples: np.ndarray, q=[0.5, 0.6, 0.7, 0.8, 0.9]) -> np.ndarray:
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
    q=[0.5, 0.6, 0.7, 0.8, 0.9],
    return_cts=False,
):
    """
    Compute class-wise quantile embeddings based on prediction scores.

    Args:
        scores_all (np.ndarray): Prediction scores. Shape (B, C) or (B,), where B is the batch size,
                                 C is the number of classes. If 1D, assumes scores for true class only.
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

    if scores_all.ndim == 2:
        for cls in range(num_classes):
            cls_scores = scores_all[labels == cls, cls]
            if cls_scores.size > 0:
                embeddings[cls] = quantile_embedding(cls_scores, q)
    else:
        for cls in range(num_classes):
            cls_scores = scores_all[labels == cls]
            if cls_scores.size > 0:
                embeddings[cls] = quantile_embedding(cls_scores, q)

    return (embeddings, cts) if return_cts else embeddings


def embed_all_tasks(
    scores_all: np.ndarray,
    labels: np.ndarray,
    q=[0.5, 0.6, 0.7, 0.8, 0.9],
    return_cts=False,
):
    """
    Compute task-wise and class-wise quantile embeddings from prediction scores.

    Args:
        scores_all (np.ndarray): Prediction scores. Shape (T, B, C) where
                                 T = number of tasks,
                                 B = batch size,
                                 C = number of classes.
        labels (np.ndarray): True class labels. Shape (B,).
        q (list): List of quantiles to compute.
        return_cts (bool): Whether to return per-task instance counts.

    Returns:
        np.ndarray: Task-wise class embeddings of shape (T, C, len(q)).
        np.ndarray (optional): Per-task instance counts. Shape (T,).
    """
    T, B, C = scores_all.shape
    quant_dim = len(q)

    embeddings = np.zeros((T, C, quant_dim))
    cts = np.zeros(T)

    for task in range(T):
        for cls in range(C):
            cls_mask = labels == cls
            cls_scores = scores_all[task, cls_mask, cls]
            if cls_scores.size > 0:
                embeddings[task, cls] = quantile_embedding(cls_scores, q)
        cts[task] = np.sum(labels < C)

    return (embeddings, cts) if return_cts else embeddings
