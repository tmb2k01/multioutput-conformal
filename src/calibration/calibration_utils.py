from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from src.calibration.clustering_utils import embed_all_classes, embed_all_tasks


def get_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute finite-sample-adjusted 1-alpha quantile of scores.

    Args:
        scores (np.ndarray): Array of conformal scores (higher = more uncertain).
        alpha (float): Miscoverage level.

    Returns:
        float: Quantile value (q-hat).
    """
    n = len(scores)
    if n == 0:
        return np.inf

    val = np.ceil((n + 1) * (1 - alpha)) / n
    if val > 1:
        return np.inf
    try:
        qhat = np.quantile(scores, val, method="inverted_cdf")
    except TypeError:
        # For older numpy versions
        qhat = np.quantile(scores, val, interpolation="lower")
    return qhat


def compute_qhat_scp_global(
    nonconformity_scores: Union[List[np.ndarray], np.ndarray],
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> float:
    """
    Compute the q-hat value for the Standard Conformal Prediction global calibration method.

    Args:
        nonconformity_scores (Union[List[np.ndarray], np.ndarray]):
            List of (B, C_t) arrays for T tasks or a single (B, C) array.
        true_labels (np.ndarray): Array of shape (T, B) or (B,) for multi-task or single-task setting.
        alpha (float): Miscoverage level.
        **kwargs: Ignored.

    Returns:
        float: Global conformal quantile (q-hat).
    """
    if isinstance(nonconformity_scores, list):
        assert (
            true_labels.ndim == 2
        ), "true_labels must be shape (T, B) for multi-task input."
        assert (
            len(nonconformity_scores) == true_labels.shape[0]
        ), "Mismatch in number of tasks."

        all_scores = []
        for task_idx, (scores_t, labels_t) in enumerate(
            zip(nonconformity_scores, true_labels)
        ):
            assert (
                scores_t.shape[0] == labels_t.shape[0]
            ), f"Mismatch in batch size for task {task_idx}."
            task_true_scores = np.take_along_axis(
                scores_t, np.expand_dims(labels_t, axis=1), axis=1
            ).squeeze(axis=1)
            all_scores.append(task_true_scores)

        concatenated_scores = np.concatenate(all_scores)
        return get_conformal_quantile(concatenated_scores, alpha)

    else:
        assert true_labels.ndim == 1, "true_labels must be 1D for single-task input."
        assert (
            nonconformity_scores.ndim == 2
        ), "nonconformity_scores must be (B, C) for single-task."
        assert (
            true_labels.shape[0] == nonconformity_scores.shape[0]
        ), "Batch size mismatch."

        true_scores = np.take_along_axis(
            nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=1
        ).squeeze(axis=1)
        return get_conformal_quantile(true_scores, alpha)


def compute_qhat_scp_task(
    nonconformity_scores: List[np.ndarray],
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> np.ndarray:
    """
    Compute the q-hat value for the Standard Conformal Prediction task-wise calibration method.

    Args:
        nonconformity_scores (List[np.ndarray]): List of nonconformity scores. Each element has shape (B, C_t).
        true_labels (np.ndarray): Array of shape (T, B) with true labels per task.
        alpha (float): Miscoverage level.
        **kwargs: Ignored.

    Returns:
        np.ndarray: Array of q-hat values, one per task. Shape (T,)
    """
    assert isinstance(
        nonconformity_scores, list
    ), "nonconformity_scores must be a list of arrays."
    assert true_labels.ndim == 2, "true_labels must be 2D (T, B)."
    assert (
        len(nonconformity_scores) == true_labels.shape[0]
    ), "Mismatch in number of tasks."

    qhats = []
    for task_idx, (scores_t, labels_t) in enumerate(
        zip(nonconformity_scores, true_labels)
    ):
        assert (
            scores_t.shape[0] == labels_t.shape[0]
        ), f"Batch size mismatch for task {task_idx}."
        true_scores = np.take_along_axis(
            scores_t, np.expand_dims(labels_t, axis=1), axis=1
        ).squeeze(axis=1)
        qhat = get_conformal_quantile(true_scores, alpha)
        qhats.append(qhat)

    return np.array(qhats)


def compute_qhat_ccp_class(
    nonconformity_scores: Union[List[np.ndarray], np.ndarray],
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Compute the q-hat values for the CCP class-wise calibration method.

    Args:
        nonconformity_scores (Union[List[np.ndarray], np.ndarray]): Nonconformity scores.
            - If np.ndarray: shape (B, C) or (B,)
            - If List[np.ndarray]: each of shape (B, C_t) or (B,) for T tasks.
        true_labels (np.ndarray): True labels, shape (B,) or (T, B).
        alpha (float): Miscoverage level.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Classwise q-hat values (or global if GT-only).
    """

    def vectorized_qhat(task_scores: np.ndarray, task_labels: np.ndarray) -> np.ndarray:
        if task_scores.ndim == 1:
            # GT-only case (B,)
            n = len(task_scores)
            k = int(np.ceil((n + 1) * (1 - alpha)))
            if k > n:
                return np.array([np.inf])
            try:
                return np.array(
                    [np.nanquantile(task_scores, k / n, method="inverted_cdf")]
                )
            except TypeError:
                return np.array(
                    [np.nanquantile(task_scores, k / n, interpolation="lower")]
                )

        B, C = task_scores.shape
        mask = task_labels[:, None] == np.arange(C)[None, :]
        selected_scores = np.where(mask, task_scores, np.nan)
        n_valid = np.sum(~np.isnan(selected_scores), axis=0)
        val = np.ceil((n_valid + 1) * (1 - alpha)) / n_valid

        qhat_per_class = np.full(C, np.inf)
        for c in range(C):
            if n_valid[c] == 0 or val[c] > 1:
                qhat_per_class[c] = np.inf
            else:
                try:
                    qhat_per_class[c] = np.nanquantile(
                        selected_scores[:, c], val[c], method="inverted_cdf"
                    )
                except TypeError:
                    qhat_per_class[c] = np.nanquantile(
                        selected_scores[:, c], val[c], interpolation="lower"
                    )
        return qhat_per_class

    if isinstance(nonconformity_scores, list):
        assert (
            true_labels.ndim == 2
        ), "Expected true_labels of shape (T, B) for list input."
        assert (
            len(nonconformity_scores) == true_labels.shape[0]
        ), "Mismatch in number of tasks."

        return [
            vectorized_qhat(scores_t, labels_t)
            for scores_t, labels_t in zip(nonconformity_scores, true_labels)
        ]

    else:
        assert isinstance(
            nonconformity_scores, np.ndarray
        ), "Expected np.ndarray or list of np.ndarray."
        assert (
            true_labels.ndim == 1
        ), "Expected true_labels of shape (B,) for single-task input."
        return vectorized_qhat(nonconformity_scores, true_labels)


def compute_qhat_ccp_task_cluster(
    nonconformity_scores: List[np.ndarray],
    true_labels: np.ndarray,
    alpha: float,
    n_clusters: int = 10,
    cluster_method: str = "kmeans",
    q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict:
    """
    Compute the q-hat values for the Clustered Conformal Prediction method (between tasks).

    Args:
        nonconformity_scores (np.ndarray): The nonconformity scores. Shape (T, B, C).
        true_labels (np.ndarray): The true labels. Shape (T, B).
        alpha (float): The miscoverage level.
        n_clusters (int): The number of clusters per task.
        cluster_method (str): Clustering method: "kmeans" or "hierarchical".
        q (Tuple[float]): Quantiles for class embeddings.

    Returns:
        dict: {
            'class_to_cluster_mapping': Dict[str, np.ndarray],
            'cluster_qhats': List[np.ndarray] (one per task)
        }
    """
    assert true_labels.ndim == 2, "Expected true_labels of shape (T, B) for list input."
    assert (
        len(nonconformity_scores) == true_labels.shape[0]
    ), "Mismatch in number of tasks."

    T = len(nonconformity_scores)
    result = {
        "class_to_cluster_mapping": {},
        "cluster_qhats": None,
    }

    # 1 - Compute class embeddings for all tasks
    task_class_embeddings = embed_all_tasks(nonconformity_scores, true_labels, q=q)

    # 2 - Cluster the class embeddings per task
    relabeled_clusters = np.zeros_like(true_labels)
    for t in range(T):
        class_embeds = task_class_embeddings[t]  # Shape: (C_t, D)
        n_classes = class_embeds.shape[0]
        n_task_clusters = min(n_clusters, n_classes)

        if cluster_method == "kmeans":
            clusterer = KMeans(n_clusters=n_task_clusters, random_state=0)
        elif cluster_method == "hierarchical":
            clusterer = AgglomerativeClustering(
                n_clusters=n_task_clusters, linkage="ward"
            )
        else:
            raise ValueError(f"Unsupported clustering method: {cluster_method}")

        cluster_labels = clusterer.fit_predict(class_embeds)  # Shape: (C_t,)
        result["class_to_cluster_mapping"][f"task-{t}"] = cluster_labels

        # Relabel each sample in the task using its class's cluster index
        relabeled_clusters[t] = cluster_labels[true_labels[t]]

    # 3 - Compute q-hats using cluster labels as pseudo-classes
    nonconformity_scores_list = [nonconformity_scores[t] for t in range(T)]
    true_cluster_labels = relabeled_clusters  # Shape (T, B)

    cluster_qhats = compute_qhat_ccp_class(
        nonconformity_scores=nonconformity_scores_list,
        true_labels=true_cluster_labels,
        alpha=alpha,
    )

    result["cluster_qhats"] = cluster_qhats
    return result


def compute_qhat_ccp_global_cluster(
    nonconformity_scores: Union[List[np.ndarray], np.ndarray],
    true_labels: np.ndarray,
    alpha: float,
    n_clusters: int = 10,
    cluster_method: str = "kmeans",
    q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> np.ndarray:
    """
    Compute the q-hat value for the Clastered Conformal Prediction method globally.
    Args:
        nonconformity_scores (Union[List[np.ndarray], np.ndarray]): Nonconformity scores.
            - If np.ndarray: shape (B, C)
            - If List[np.ndarray]: each of shape (B, C_t) for T tasks.
        true_labels (np.ndarray): The true labels. Shape (B,) or (T, B).
        alpha (float): The miscoverage level.
        n_clusters (int): The number of clusters.
        cluster_method (str): The clustering method to use.
                              Options: "kmeans", "hierarchical"
        q (list): The quantiles to compute for each class.
                              Default: [0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        np.ndarray: The classwise q-hat values of shape (C,) or shape (T,C,) for each task.
    """

    def cluster_embeddings(
        embeds: np.ndarray, n_clusters: int, method: str
    ) -> np.ndarray:
        """
        Cluster the embeddings using the specified method.
        Args:
            embeds (np.ndarray): The embeddings to cluster.
            n_clusters (int): The number of clusters.
            method (str): The clustering method to use.
                          Options: "kmeans", "hierarchical"
        Returns:
            np.ndarray: The cluster labels for each embedding.
        """
        min_clusters = min(n_clusters, embeds.shape[0])
        if method == "kmeans":
            return KMeans(n_clusters=min_clusters, random_state=0).fit_predict(embeds)
        elif method == "hierarchical":
            return AgglomerativeClustering(
                n_clusters=min_clusters, affinity="euclidean", linkage="ward"
            ).fit_predict(embeds)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

    result = {
        "class_to_cluster_mapping": {},
        "cluster_qhats": None,
    }

    if isinstance(nonconformity_scores, list):
        # If nonconformity_scores is a list (multi-task)
        is_multitask = True
        assert (
            len(nonconformity_scores) == true_labels.shape[0]
        ), "Mismatch between the number of tasks and true labels."
    else:
        # If nonconformity_scores is a single array (single-task)
        is_multitask = False
        assert (
            nonconformity_scores.shape[0] == true_labels.shape[0]
        ), "Mismatch between the batch size and true labels."

    if is_multitask:
        # Get class embeddings per task
        class_embeddings = embed_all_tasks(
            nonconformity_scores, true_labels, q=q
        )  # List[np.ndarray] of shape (C_t, D)
        num_classes_per_task = [emb.shape[0] for emb in class_embeddings]
        flat_embeddings = np.vstack(class_embeddings)  # Shape: (sum(C_t), D)

        # Cluster all class embeddings
        flat_cluster_labels = cluster_embeddings(
            flat_embeddings, n_clusters, cluster_method
        )

        # Re-split cluster labels by task
        split_indices = np.cumsum(num_classes_per_task)[:-1]
        per_task_cluster_labels = np.split(flat_cluster_labels, split_indices)

        for t, labels in enumerate(per_task_cluster_labels):
            result["class_to_cluster_mapping"][f"task-{t}"] = labels

        # Relabel clusters using true_labels per task
        flattened_scores = []
        relabeled_clusters = []

        for t, (scores_t, labels_t) in enumerate(
            zip(nonconformity_scores, per_task_cluster_labels)
        ):
            B_t = scores_t.shape[0]
            true_labels_t = true_labels[t]  # shape (B_t,)

            # Get nonconformity score for the true class
            task_scores = scores_t[np.arange(B_t), true_labels_t]  # shape (B_t,)
            flattened_scores.append(task_scores)

            # Map true class to cluster label
            task_clusters = labels_t[true_labels_t]
            relabeled_clusters.append(task_clusters)

        all_scores = np.concatenate(flattened_scores)  # shape (total B,)
        flat_labels = np.concatenate(relabeled_clusters)  # shape (total B,)

    else:
        class_embeddings = embed_all_classes(nonconformity_scores, true_labels, q=q)
        cluster_labels = cluster_embeddings(
            class_embeddings, n_clusters, cluster_method
        )
        result["class_to_cluster_mapping"] = cluster_labels
        flat_labels = cluster_labels[true_labels]
        all_scores = nonconformity_scores

    result["cluster_qhats"] = compute_qhat_ccp_class(all_scores, flat_labels, alpha)

    return result


CALIBRATION_FN_HIGH_DIC: Dict[str, Callable[..., float]] = {
    "scp_global_threshold": compute_qhat_scp_global,
    "scp_task_thresholds": compute_qhat_scp_task,
    "ccp_class_thresholds": compute_qhat_ccp_class,
    "ccp_task_cluster_thresholds": compute_qhat_ccp_task_cluster,
    "ccp_global_cluster_thresholds": compute_qhat_ccp_global_cluster,
}

CALIBRATION_FN_LOW_DIC: Dict[str, Callable[..., float]] = {
    "scp_global_threshold": compute_qhat_scp_global,
    "ccp_class_thresholds": compute_qhat_ccp_class,
    "ccp_global_clusters": compute_qhat_ccp_global_cluster,
}
