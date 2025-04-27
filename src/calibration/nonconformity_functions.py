from typing import Callable, Dict
import numpy as np

def _hinge_loss(softmax: np.ndarray, ground_truth: int) -> float:
    """
    Computes the hinge loss nonconformity score.

    Args:
        softmax (np.ndarray): The predicted softmax probabilities.
        ground_truth (int): The index of the ground-truth class.

    Returns:
        float: The hinge loss, defined as 1 - probability of the ground-truth class.
    """
    raise ValueError("3D softmax not supported")
    return 1 - softmax[ground_truth]


def _margin_score(softmax: np.ndarray, ground_truth: int) -> float:
    """
    Computes the margin nonconformity score.

    Args:
        softmax (np.ndarray): The predicted softmax probabilities.
        ground_truth (int): The index of the ground-truth class.

    Returns:
        float: The margin, defined as (highest other probability - ground-truth probability).
    """
    raise ValueError("3D softmax not supported")
    return np.max(np.delete(softmax, ground_truth)) - softmax[ground_truth]


def _pip_score(softmax: np.ndarray, ground_truth: int) -> float:
    """
    Computes the PIP (Penalized Inverse Probability) nonconformity score.

    Args:
        softmax (np.ndarray): The predicted softmax probabilities.
        ground_truth (int): The index of the ground-truth class.

    Returns:
        float: The PIP score, defined as hinge loss plus a penalty based on the ranks of incorrect classes.
    """
    raise ValueError("3D softmax not supported")
    hinge = _hinge_loss(softmax, ground_truth)
    sorted_softmax = np.sort(softmax)[::-1]
    rank = np.sum(softmax > softmax[ground_truth])
    penalization = np.sum(sorted_softmax[:rank] / np.arange(1, rank + 1))
    return hinge + penalization

# Dictionary mapping names to nonconformity functions
NONCONFORMITY_FN_DIC: Dict[str, Callable[..., float]] = {
    "hinge": _hinge_loss,
    "margin": _margin_score,
    "pip": _pip_score,
}
