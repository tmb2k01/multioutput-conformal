from typing import Callable, Dict
import numpy as np


def _hinge_loss(softmax: np.ndarray) -> np.ndarray:
    """
    Computes the hinge loss nonconformity score for all classes.

    The hinge loss for a class is defined as 1 minus the predicted probability assigned to that class.

    Args:
        softmax (np.ndarray): Predicted softmax probabilities.
            - Shape (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.

    Returns:
        np.ndarray: Hinge loss values for each class.
            - Shape (T, B, C) for (T, B, C) input.
            - Shape (B, C) for (B, C) input.
    """
    return 1 - softmax


def _margin_score(softmax: np.ndarray) -> np.ndarray:
    """
    Computes the margin nonconformity score for all classes.

    The margin for each class is defined as (highest incorrect class probability - class probability).

    Args:
        softmax (np.ndarray): Predicted softmax probabilities.
            - Shape (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.

    Returns:
        np.ndarray: Margin scores for each class.
            - Shape (T, B, C) for (T, B, C) input.
            - Shape (B, C) for (B, C) input.
    """

    def compute_margin(softmax_batch: np.ndarray) -> np.ndarray:
        """
        Computes margin score for each class in the batch.

        Args:
            softmax_batch (np.ndarray): Softmax probabilities, shape (B, C).

        Returns:
            np.ndarray: Margin scores, shape (B, C).
        """
        highest_other = np.max(
            np.delete(softmax_batch, np.arange(softmax_batch.shape[1]), axis=1), axis=1
        )
        return highest_other[:, None] - softmax_batch

    if len(softmax.shape) == 3:
        return np.array([compute_margin(softmax[i]) for i in range(softmax.shape[0])])
    elif len(softmax.shape) == 2:
        return compute_margin(softmax)
    else:
        raise ValueError(
            f"Expected softmax shape (B, C) or (T, B, C), got {softmax.shape}"
        )


def _pip_score(softmax: np.ndarray) -> np.ndarray:
    """
    Computes the PIP (Penalized Inverse Probability) nonconformity score for all classes.

    The PIP score for each class is the hinge loss plus a penalization based on the ranks of incorrect classes.

    Args:
        softmax (np.ndarray): Predicted softmax probabilities.
            - Shape (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.

    Returns:
        np.ndarray: PIP scores for each class.
            - Shape (T, B, C) for (T, B, C) input.
            - Shape (B, C) for (B, C) input.
    """

    def compute_pip(softmax_batch: np.ndarray) -> np.ndarray:
        """
        Computes PIP score for each class in the batch.

        Args:
            softmax_batch (np.ndarray): Softmax probabilities, shape (B, C).

        Returns:
            np.ndarray: PIP scores, shape (B, C).
        """
        hinge = 1 - softmax_batch
        sorted_softmax = np.sort(softmax_batch, axis=1)[:, ::-1]
        penalizations = np.zeros_like(hinge)
        for i in range(softmax_batch.shape[0]):
            for rank in range(1, softmax_batch.shape[1]):
                penalizations[i, rank] = np.sum(
                    sorted_softmax[i, :rank] / np.arange(1, rank + 1)
                )

        return hinge + penalizations

    if len(softmax.shape) == 3:
        return np.array([compute_pip(softmax[i]) for i in range(softmax.shape[0])])
    elif len(softmax.shape) == 2:
        return compute_pip(softmax)
    else:
        raise ValueError(
            f"Expected softmax shape (B, C) or (T, B, C), got {softmax.shape}"
        )


# Dictionary mapping names to nonconformity functions
NONCONFORMITY_FN_DIC: Dict[str, Callable[..., np.ndarray]] = {
    "hinge": _hinge_loss,
    "margin": _margin_score,
    "pip": _pip_score,
}
