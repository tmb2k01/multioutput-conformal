from typing import Callable, Dict
import numpy as np


def _hinge_loss(softmax: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Computes the hinge loss nonconformity score.

    The hinge loss is defined as 1 minus the predicted probability assigned to the ground-truth class.

    Args:
        softmax (np.ndarray): Predicted softmax probabilities.
            - Shape (T, B, C) if multiple tasks.
            - Shape (B, C) if single task.
        ground_truth (np.ndarray): Ground-truth class indices.
            - Shape (T, B) if multiple tasks.
            - Shape (B,) if single task.

    Returns:
        np.ndarray: Hinge loss values.
            - Shape (T, B) if input was (T, B, C).
            - Shape (B,) if input was (B, C).
    """

    def compute_hinge(
        softmax_batch: np.ndarray, ground_truth_batch: np.ndarray
    ) -> np.ndarray:
        """
        Computes hinge loss for a batch.

        Args:
            softmax_batch (np.ndarray): Softmax probabilities, shape (B, C).
            ground_truth_batch (np.ndarray): Ground-truth indices, shape (B,).

        Returns:
            np.ndarray: Hinge losses, shape (B,).
        """
        return 1 - softmax_batch[np.arange(softmax_batch.shape[0]), ground_truth_batch]

    if len(softmax.shape) == 3:
        # Multiple tasks: loop over T
        return np.array(
            [
                compute_hinge(softmax[i], ground_truth[i])
                for i in range(softmax.shape[0])
            ]
        )
    elif len(softmax.shape) == 2:
        # Single task
        return compute_hinge(softmax, ground_truth)
    else:
        raise ValueError(
            f"Expected softmax shape (B, C) or (T, B, C), got {softmax.shape}"
        )


def _margin_score(softmax: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Computes the margin nonconformity score.

    The margin is defined as (highest incorrect class probability - ground-truth class probability).

    Args:
        softmax (np.ndarray): Predicted softmax probabilities.
            - Shape (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.
        ground_truth (np.ndarray): Ground-truth class indices.
            - Shape (T, B) or (B,).

    Returns:
        np.ndarray: Margin scores.
            - Shape (T, B) if input was (T, B, C).
            - Shape (B,) if input was (B, C).
    """

    def compute_margin(
        softmax_batch: np.ndarray, ground_truth_batch: np.ndarray
    ) -> np.ndarray:
        """
        Computes margin score for a batch.

        Args:
            softmax_batch (np.ndarray): Softmax probabilities, shape (B, C).
            ground_truth_batch (np.ndarray): Ground-truth indices, shape (B,).

        Returns:
            np.ndarray: Margin scores, shape (B,).
        """
        # Mask the ground-truth entries by setting them to -inf
        mask = np.ones_like(softmax_batch, dtype=bool)
        mask[np.arange(softmax_batch.shape[0]), ground_truth_batch] = False

        masked_softmax = np.where(mask, softmax_batch, -np.inf)
        highest_other = np.max(masked_softmax, axis=1)
        ground_truth_probs = softmax_batch[
            np.arange(softmax_batch.shape[0]), ground_truth_batch
        ]

        return highest_other - ground_truth_probs

    if len(softmax.shape) == 3:
        return np.array(
            [
                compute_margin(softmax[i], ground_truth[i])
                for i in range(softmax.shape[0])
            ]
        )
    elif len(softmax.shape) == 2:
        return compute_margin(softmax, ground_truth)
    else:
        raise ValueError(
            f"Expected softmax shape (B, C) or (T, B, C), got {softmax.shape}"
        )


def _pip_score(softmax: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Computes the PIP (Penalized Inverse Probability) nonconformity score.

    The PIP score is the hinge loss plus a penalization based on the ranks of incorrect classes.

    Args:
        softmax (np.ndarray): Predicted softmax probabilities.
            - Shape (T, B, C) for multiple tasks.
            - Shape (B, C) for a single task.
        ground_truth (np.ndarray): Ground-truth class indices.
            - Shape (T, B) or (B,).

    Returns:
        np.ndarray: PIP scores.
            - Shape (T, B) if input was (T, B, C).
            - Shape (B,) if input was (B, C).
    """

    def compute_pip(
        softmax_batch: np.ndarray, ground_truth_batch: np.ndarray
    ) -> np.ndarray:
        """
        Computes PIP score for a batch.

        Args:
            softmax_batch (np.ndarray): Softmax probabilities, shape (B, C).
            ground_truth_batch (np.ndarray): Ground-truth indices, shape (B,).

        Returns:
            np.ndarray: PIP scores, shape (B,).
        """
        batch_size = softmax_batch.shape[0]

        # Hinge part
        hinge = 1 - softmax_batch[np.arange(batch_size), ground_truth_batch]

        # Sort softmax descending per sample
        sorted_softmax = np.sort(softmax_batch, axis=1)[:, ::-1]

        # Find how many classes have a higher probability than the ground-truth
        gt_probs = softmax_batch[np.arange(batch_size), ground_truth_batch][:, None]
        ranks = np.sum(softmax_batch > gt_probs, axis=1)

        # Compute penalization term
        penalizations = np.zeros(batch_size)
        for i in range(batch_size):
            if ranks[i] > 0:
                penalizations[i] = np.sum(
                    sorted_softmax[i, : ranks[i]] / np.arange(1, ranks[i] + 1)
                )

        return hinge + penalizations

    if len(softmax.shape) == 3:
        return np.array(
            [compute_pip(softmax[i], ground_truth[i]) for i in range(softmax.shape[0])]
        )
    elif len(softmax.shape) == 2:
        return compute_pip(softmax, ground_truth)
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
