from typing import Callable, Counter, Dict, List, Tuple, Union

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from src.calibration.clustering_utils import embed_all_classes, embed_all_tasks, get_clustering_parameters

def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n

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
        sorted_scores = np.sort(scores)
        qhat = np.quantile(sorted_scores, val, interpolation="lower")
    return qhat


def compute_qhat_scp_global(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> float:
    """
    Compute the q-hat value for the Standard Conformal Prediction global calibration method.

    Args:
        nonconformity_scores (np.ndarray): Nonconformity scores of the ground truth labels with shape
                                           (T, B) or (B,) for multi-task or single-task setting.
        true_labels (np.ndarray): Ground truth labels with shape (T, B) or (B,) for multi-task or single-task setting.
        alpha (float): Miscoverage level.
        **kwargs: Ignored.

    Returns:
        float: Global conformal quantile (q-hat).
    """
    assert (
        nonconformity_scores.shape == true_labels.shape
    ), "Mismatch in shape between nonconformity scores and true labels."
    assert nonconformity_scores.ndim in (
        1,
        2,
    ), "Input dimension error - nonconformity_scores must be 1D (B,) or 2D (T, B)."

    if nonconformity_scores.ndim == 2:
        all_scores = nonconformity_scores.reshape(-1)
    else:
        all_scores = nonconformity_scores

    return get_conformal_quantile(all_scores, alpha)


def compute_qhat_scp_task(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> np.ndarray:
    """
    Compute the q-hat value for the Standard Conformal Prediction task-wise calibration method.

    Args:
        nonconformity_scores (np.ndarray): Nonconformity scores of the ground truth labels with shape (T, B).
        true_labels (np.ndarray): Ground truth labels with shape (T, B) with true labels per task.
        alpha (float): Miscoverage level.
        **kwargs: Ignored.

    Returns:
        np.ndarray: Array of q-hat values, one per task. Shape (T,)
    """
    assert (
        nonconformity_scores.shape == true_labels.shape
    ), "Mismatch in shape between nonconformity scores and true labels."
    assert (
        nonconformity_scores.ndim == 2
    ), "Input dimension error - nonconformity_scores must be 2D (T, B)."

    return np.apply_along_axis(get_conformal_quantile, 1, nonconformity_scores, alpha)


def compute_qhat_ccp_class(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    **kwargs,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Compute the q-hat values for the CCP class-wise calibration method.

    Args:
        nonconformity_scores (np.ndarray): Nonconformity scores of the ground truth labels with shape
                                           (T, B) or (B,) for multi-task or single-task setting.
        true_labels (np.ndarray): Ground truth labels with shape (T, B) or (B,) for multi-task or single-task setting.
        alpha (float): Miscoverage level.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Classwise q-hat values. Shape (T, C) for multi-task or (C,) for single-task.
    """

    def vectorized_qhat(task_scores: np.ndarray, task_labels: np.ndarray) -> np.ndarray:
        """
        Compute q-hat for a single task using vectorized operations.

        Args:
            task_scores (np.ndarray): Nonconformity scores for the task.
            task_labels (np.ndarray): True labels for the task.

        Returns:
            np.ndarray: q-hat values for the task.
        """
        unique_classes = np.unique(task_labels)
        n_classes = unique_classes.max() + 1
        qhats = np.full(n_classes, np.inf)

        for c in unique_classes:
            class_mask = task_labels == c
            class_scores = task_scores[class_mask]
            if class_scores.size > 0:
                qhats[c] = get_conformal_quantile(class_scores, alpha)

        return qhats

    assert (
        nonconformity_scores.shape == true_labels.shape
    ), "Mismatch in shape between nonconformity scores and true labels."
    assert nonconformity_scores.ndim in (
        1,
        2,
    ), "Input dimension error - nonconformity_scores must be 1D (B,) or 2D (T, B)."

    if nonconformity_scores.ndim == 2:
        return [
            vectorized_qhat(scores_t, labels_t)
            for scores_t, labels_t in zip(nonconformity_scores, true_labels)
        ]

    else:
        return vectorized_qhat(nonconformity_scores, true_labels)


def compute_qhat_ccp_task_cluster(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    n_clusters: Union[int, str] = "auto",
    cluster_method: str = "kmeans",
    q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict:
    """
    Compute q-hat values per task using Clustered Conformal Prediction (CCP).

    Args:
        nonconformity_scores (np.ndarray): Nonconformity scores of the ground truth labels with shape (T, B).
        true_labels (np.ndarray): Ground truth labels with shape (T, B) with true labels per task.
        alpha (float): The miscoverage level.
        n_clusters (int): The number of clusters per task.
        cluster_method (str): Clustering method: "kmeans" or "hierarchical".
        q (Tuple[float]): Quantiles for class embeddings.

    Returns:
        dict: {
            'task-0': {'mapping': np.ndarray, 'qhats': np.ndarray},
            'task-1': {...},
            ...
        }
    """
    assert true_labels.ndim == 2, "Expected true_labels of shape (T, B)"
    assert (
        nonconformity_scores.shape == true_labels.shape
    ), "Shape mismatch between scores and labels."

    T = nonconformity_scores.shape[0]
    result = {}

    # 1 - Embed each class per task
    task_class_embeddings = embed_all_tasks(nonconformity_scores, true_labels, q=q)

    # 1.5 - Auto-select number of clusters if needed
    if n_clusters == "auto":
        num_clusters_per_task = []

        for task_id, row in enumerate(true_labels):

            # Count frequencies for this task only
            cts_dict = Counter(row)

            # Ensure all classes up to max(row) are included
            n_classes = np.max(row) + 1
            cts = [cts_dict.get(k, 0) for k in range(n_classes)]

            n_min = min(cts)
            n_thresh = get_quantile_threshold(alpha)
            n_min = max(n_min, n_thresh)  # exclude classes with too few examples
            num_remaining_classes = np.sum(np.array(cts) >= n_min)

            # Compute clustering params *per task*
            n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)

            print(f"[Task {task_id}] n_clustering={n_clustering}, "
                f"num_clusters={num_clusters}")

            num_clusters_per_task.append(num_clusters)

    # 2 - For each task, perform clustering and compute q-hats
    for t in range(T):
        class_embeds = task_class_embeddings[t]  # (C_t, D)
        n_classes = class_embeds.shape[0]
        n_task_clusters = min(num_clusters_per_task[t], n_classes)

        # Choose clustering method
        if cluster_method == "kmeans":
            clusterer = KMeans(n_clusters=n_task_clusters, random_state=0)
        elif cluster_method == "hierarchical":
            clusterer = AgglomerativeClustering(
                n_clusters=n_task_clusters, linkage="ward"
            )
        else:
            raise ValueError(f"Unsupported clustering method: {cluster_method}")

        # Cluster class embeddings
        cluster_labels = clusterer.fit_predict(class_embeds)  # (C_t,)
        relabeled = cluster_labels[true_labels[t]]  # Map B labels to cluster indices

        # Compute q-hats for this task
        task_qhats = compute_qhat_ccp_class(
            nonconformity_scores=nonconformity_scores[t],
            true_labels=relabeled,
            alpha=alpha,
        )

        result[f"task-{t}"] = {
            "mapping": cluster_labels,
            "qhats": task_qhats,
        }

    return result


def compute_qhat_ccp_global_cluster(
    nonconformity_scores: np.ndarray,
    true_labels: np.ndarray,
    alpha: float,
    n_clusters: Union[int, str] = "auto",
    cluster_method: str = "kmeans",
    q: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict:
    """
    Compute the q-hat value for the Clastered Conformal Prediction method globally.
    Args:
        nonconformity_scores (np.ndarray): Nonconformity scores of the ground truth labels with shape
                                           (T, B) or (B,) for multi-task or single-task setting.
        true_labels (np.ndarray): Ground truth labels with shape (T, B) or (B,) for multi-task or single-task setting.
        alpha (float): The miscoverage level.
        n_clusters (int): The number of clusters.
        cluster_method (str): The clustering method to use.
                              Options: "kmeans", "hierarchical"
        q (list): The quantiles to compute for each class.
                              Default: [0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        dict: {
            'class_to_cluster_mapping': Dict[str, np.ndarray],
            'cluster_qhats': np.ndarray
        }
    """
    assert nonconformity_scores.shape == true_labels.shape, "Shape mismatch."
    assert nonconformity_scores.ndim in (1, 2), "Expected 1D or 2D inputs."

    is_multitask = nonconformity_scores.ndim == 2
    result = {"class_to_cluster_mapping": {}, "cluster_qhats": None}

    # Step 1: Embed class-level representations
    if is_multitask:
        embeddings = embed_all_tasks(
            nonconformity_scores, true_labels, q=q
        )  # List of (C_t, D)
        num_classes_per_task = [emb.shape[0] for emb in embeddings]
        flat_embeddings = np.vstack(embeddings)  # (total_C, D)
    else:
        flat_embeddings = embed_all_classes(
            nonconformity_scores, true_labels, q=q
        )  # (C, D)
        num_classes_per_task = [flat_embeddings.shape[0]]


    # 1.5 - Auto-select number of clusters if needed
    if n_clusters == "auto":
        true_labels_cluster = true_labels
        if true_labels.ndim == 1:
            true_labels_cluster = true_labels_cluster[np.newaxis, :]  # shape (1, N)
        
        # Flatten to tuples (row_id, label)
        task_label_pairs = [(i, lbl) for i, row in enumerate(true_labels_cluster) for lbl in row]
        cts_dict = Counter(task_label_pairs)

        # Suppose num_classes_per_task = [max label count for each row]
        num_classes_per_task = [np.max(row) + 1 for row in true_labels_cluster]

        cts = []
        for i, n_classes in enumerate(num_classes_per_task):
            for k in range(n_classes):
                cts.append(cts_dict.get((i, k), 0))

        n_min = min(cts)
        n_thresh = get_quantile_threshold(alpha) 
        n_min = max(n_min, n_thresh)  # Classes with fewer than n_thresh examples will be excluded
        num_remaining_classes = np.sum(np.array(cts) >= n_min)

        n_clustering, n_clusters = get_clustering_parameters(num_remaining_classes, n_min)
        print(f'n_clustering={n_clustering}, num_clusters={n_clusters}')

    # Step 2: Choose and apply clustering method
    n_classes = flat_embeddings.shape[0]
    n_task_clusters = min(n_clusters, n_classes)

    clusterer = {
        "kmeans": KMeans(n_clusters=n_task_clusters, random_state=0),
        "hierarchical": AgglomerativeClustering(
            n_clusters=n_task_clusters, linkage="ward"
        ),
    }[cluster_method]

    flat_cluster_labels = clusterer.fit_predict(flat_embeddings)

    # Step 3: Map cluster labels back to tasks/classes
    if is_multitask:
        split_indices = np.cumsum(num_classes_per_task)[:-1]
        per_task_labels = np.split(flat_cluster_labels, split_indices)

        for t, labels in enumerate(per_task_labels):
            result["class_to_cluster_mapping"][f"task-{t}"] = labels

        # Build relabeled flat cluster assignments
        relabeled_clusters = np.stack(
            [per_task_labels[t][true_labels[t]] for t in range(len(per_task_labels))]
        )
        flat_labels = relabeled_clusters.reshape(-1)
        all_scores = nonconformity_scores.reshape(-1)
    else:
        result["class_to_cluster_mapping"] = flat_cluster_labels
        flat_labels = flat_cluster_labels[true_labels]
        all_scores = nonconformity_scores

    # Step 4: Compute q-hats by cluster
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
