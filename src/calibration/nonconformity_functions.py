from typing import Callable, Dict

import numpy as np


def _hinge_loss(softmax: np.ndarray) -> np.ndarray:
    """
    Computes the hinge loss nonconformity score for all classes.

    The hinge loss for a class is defined as 1 minus the predicted probability assigned to that class.

    Args:
        softmax (np.ndarray): Predicted softmax probabilities. Shape (B, C).

    Returns:
        np.ndarray: Hinge loss values for each class. Shape (B, C).
    """
    return 1 - softmax


def _margin_score(softmax: np.ndarray) -> np.ndarray:
    """
    Computes the margin nonconformity score for all classes.

    The margin for each class is defined as (highest incorrect class probability - class probability).

    Args:
        softmax (np.ndarray): Predicted softmax probabilities. Shape (B, C).

    Returns:
        np.ndarray: Margin scores for each class. Shape (B, C).
    """
    B, C = softmax.shape
    margins = np.empty((B, C))
    for j in range(C):
        mask = np.ones(C, dtype=bool)
        mask[j] = False
        margins[:, j] = np.max(softmax[:, mask], axis=1) - softmax[:, j]
    return margins


def _pip_score(softmax: np.ndarray) -> np.ndarray:
    """
    Computes the PIP (Penalized Inverse Probability) nonconformity score for all classes.

    The PIP score for each class is the hinge loss plus a penalization based on the ranks of incorrect classes.

    Args:
        softmax (np.ndarray): Predicted softmax probabilities. Shape (B, C).

    Returns:
        np.ndarray: PIP scores for each class.Shape (B, C).
    """
    hinge = 1 - softmax
    sorted_softmax = np.sort(softmax, axis=1)[:, ::-1]
    penalizations = np.zeros_like(hinge)
    for i in range(softmax.shape[0]):
        for rank in range(1, softmax.shape[1]):
            penalizations[i, rank] = np.sum(
                sorted_softmax[i, :rank] / np.arange(1, rank + 1)
            )

    return hinge + penalizations


# Dictionary mapping names to nonconformity functions
NONCONFORMITY_FN_DIC: Dict[str, Callable[..., np.ndarray]] = {
    "hinge": _hinge_loss,
    "margin": _margin_score,
    "pip": _pip_score,
}
