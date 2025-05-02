import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import Callable, Dict

from src.calibration.clustering_utils import embed_all_tasks, embed_all_classes


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
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
):
    """
    Compute the q-hat value for the Standard Conformal Prediction global calibration method.

    Args:
        nonconformity_scores (np.ndarray): The nonconformity scores. Shape (B, C) or (T, B, C).
        true_labels (np.ndarray): The true labels. Shape (B,) or (T, B).
        alpha (float): The miscoverage level.
        **kwargs (dict): Ignored.

    Returns:
        float: The global standard q-hat value.
    """
    assert nonconformity_scores.ndim in (
        2,
        3,
    ), "Nonconformity scores should be 2D or 3D."
    assert (
        true_labels.shape[0] == nonconformity_scores.shape[0]
    ), "Mismatch between true_labels and nonconformity_scores dimensions."

    if len(nonconformity_scores.shape) == 3:
        assert (
            true_labels.shape[1] == nonconformity_scores.shape[1]
        ), "Mismatch between true_labels and nonconformity_scores dimensions."
        true_scores = np.take_along_axis(
            nonconformity_scores, np.expand_dims(true_labels, axis=2), axis=2
        ).squeeze(axis=2)
        return get_conformal_quantile(true_scores.flatten(), alpha)

    elif len(nonconformity_scores.shape) == 2:
        true_scores = np.take_along_axis(
            nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=1
        ).squeeze(axis=1)
        return get_conformal_quantile(true_scores, alpha)


def compute_qhat_scp_task(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
):
    """
    Compute the q-hat value for the Standard Conformal Prediction task calibration method.

    Args:
        nonconformity_scores (np.ndarray): The nonconformity scores. Shape (T, B, C).
        true_labels (np.ndarray): The true labels. Shape (T, B).
        alpha (float): The miscoverage level.
        **kwargs (dict): Ignored.

    Returns:
        np.ndarray: The standard q-hat values for each task.
    """
    assert nonconformity_scores.ndim == 3, "Nonconformity scores should be 3D."
    assert (
        true_labels.shape == nonconformity_scores.shape[:2]
    ), "Shape mismatch between true_labels and nonconformity_scores."

    true_scores = np.take_along_axis(
        nonconformity_scores, np.expand_dims(true_labels, axis=1), axis=2
    ).squeeze(axis=2)
    return np.apply_along_axis(get_conformal_quantile, 1, true_scores, alpha)


def compute_qhat_ccp_class(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> np.ndarray:
    """
    Compute the q-hat value for the CCP class calibration method.
    Args:
        nonconformity_scores (np.ndarray): The nonconformity scores. Shape (B, C) or (T, B, C).
        true_labels (np.ndarray): The true labels. Shape (B,) or (T, B).
        alpha (float): The miscoverage level.
        **kwargs (dict): Ignored.

    Returns:
        np.ndarray: The classwise q-hat values of shape (C,) or shape (T,C,) for each task.
    """

    def vectorized_qhat(task_scores, task_labels, num_classes):
        """
        Compute class-wise quantiles for given task scores and labels.

        Args:
            task_scores (np.ndarray): Nonconformity scores of shape (B, C).
            task_labels (np.ndarray): True labels of shape (B,).
            num_classes (int): Number of classes.

        Returns:
            np.ndarray: The q-hat values of shape (C,).
        """
        # task_scores: (B, C), task_labels: (B,)
        # Create (B, C) mask where mask[b, c] = True if label[b] == c
        mask = task_labels[:, None] == np.arange(num_classes)[None, :]  # (B, C)

        # Set scores to nan where not matching the class
        selected_scores = np.where(mask, task_scores, np.nan)  # (B, C)

        # Now compute quantiles ignoring nan
        n_valid = np.sum(~np.isnan(selected_scores), axis=0)  # (C,)
        val = np.ceil((n_valid + 1) * (1 - alpha)) / n_valid  # (C,)

        # Compute quantiles per class
        qhat_per_class = np.full(num_classes, np.inf)
        for c in range(num_classes):
            if n_valid[c] == 0 or val[c] > 1:
                qhat_per_class[c] = np.inf
            else:
                class_scores = selected_scores[:, c]
                qhat_per_class[c] = np.nanquantile(
                    class_scores, val[c], method="inverted_cdf"
                )

        return qhat_per_class

    assert nonconformity_scores.ndim in (
        2,
        3,
    ), "Expected nonconformity_scores of shape (B, C) or (T, B, C)"

    if nonconformity_scores.ndim == 3:
        T, B, C = nonconformity_scores.shape
        qhats = np.zeros((T, C))
        for t in range(T):
            qhats[t] = vectorized_qhat(nonconformity_scores[t], true_labels[t], C)
        return qhats
    else:
        B, C = nonconformity_scores.shape
        return vectorized_qhat(nonconformity_scores, true_labels, C)


def compute_qhat_ccp_task_cluster(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    n_clusters: int = 10,
    cluster_method: str = "kmeans",
    q: list = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> np.ndarray:
    """
    Compute the q-hat value for the Clustered Conformal Prediction method between tasks.
    Args:
        nonconformity_scores (np.ndarray): The nonconformity scores. Shape (T, B, C).
        true_labels (np.ndarray): The true labels. Shape (T, B).
        alpha (float): The miscoverage level.
        n_clusters (int): The number of clusters.
        cluster_method (str): The clustering method to use.
                              Options: "kmeans", "hierarchical"
        q (list): The quantiles to compute for each class.
                              Default: [0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        np.ndarray: The classwise q-hat values of shape (T,C,) for each task.
    """

    assert (
        nonconformity_scores.ndim == 3
    ), "Expected nonconformity_scores of shape (T, B, C)"

    assert (
        true_labels.shape == nonconformity_scores.shape[:2]
    ), "Shape mismatch between true_labels and nonconformity_scores."

    # 0 - Initialize the result dictionary
    result = {
        "class_to_cluster_mapping": {},
        "cluster_qhats": None,
    }

    # 1 - Compute classes embeddings
    task_class_embeddings = embed_all_tasks(nonconformity_scores, true_labels, q=q)

    # 2 - Cluster the embeddings
    relabeled_clusters = np.zeros_like(true_labels)
    for t in range(task_class_embeddings.shape[0]):
        class_embeds = task_class_embeddings[t]
        min_n_clusters = min(n_clusters, class_embeds.shape[0])

        if cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=min_n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(class_embeds)
        elif cluster_method == "hierarchical":
            hierarchical = AgglomerativeClustering(
                n_clusters=min_n_clusters, affinity="euclidean", linkage="ward"
            )
            cluster_labels = hierarchical.fit_predict(class_embeds)
        else:
            raise ValueError(f"Unsupported clustering method: {cluster_method}")

        result["class_to_cluster_mapping"][f"task-{t}"] = cluster_labels

        # Relabel true labels using cluster assignments
        relabeled_clusters[t] = cluster_labels[true_labels[t]]

    # 3 - Compute the q-hat values
    result["cluster_qhats"] = compute_qhat_ccp_class(
        nonconformity_scores, relabeled_clusters, alpha
    )
    return result


def compute_qhat_ccp_global_cluster(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    n_clusters: int = 10,
    cluster_method: str = "kmeans",
    q: list = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> np.ndarray:
    """
    Compute the q-hat value for the Clastered Conformal Prediction method globally.
    Args:
        nonconformity_scores (np.ndarray): The nonconformity scores. Shape (B, C) or (T, B, C).
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
    assert nonconformity_scores.ndim in (
        2,
        3,
    ), "Expected nonconformity_scores of shape (B, C) or (T, B, C)"
    assert (
        true_labels.shape == nonconformity_scores.shape[:2]
    ), "Shape mismatch between true_labels and nonconformity_scores."

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

    is_multitask = nonconformity_scores.ndim == 3

    class_embeddings = (
        embed_all_tasks(nonconformity_scores, true_labels, q=q)
        if is_multitask
        else embed_all_classes(nonconformity_scores, true_labels, q=q)
    )

    # Flatten embeddings and cluster
    if is_multitask:
        T, C, D = class_embeddings.shape
        flat_embeddings = class_embeddings.reshape(-1, D)
        cluster_labels = cluster_embeddings(flat_embeddings, n_clusters, cluster_method)
        cluster_labels = cluster_labels.reshape(T, C)

        for t in range(T):
            result["class_to_cluster_mapping"][f"task-{t}"] = cluster_labels[t]

        relabeled_clusters = np.array(
            [cluster_labels[t][true_labels[t]] for t in range(T)]
        )

        all_scores = nonconformity_scores.reshape(-1, nonconformity_scores.shape[-1])
        flat_labels = relabeled_clusters.flatten()
    else:
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
