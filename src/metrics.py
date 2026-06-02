
import numpy as np


def compute_informativeness(predictions: list[np.ndarray] | list[list[np.ndarray]]) -> float:
    """
    Informativeness = fraction of samples where the prediction set is singleton.

    

    Args:
        predictions (List[np.ndarray]): A list of predictions.

    Returns:
        float:  
            Single-task:
                A sample is informative if its prediction array has size 1.

            Multi-task:
                A sample is informative if every task prediction array has size 1.
    """
    if isinstance(predictions[0], np.ndarray):
        return np.mean([pred.size == 1 for pred in predictions])

    return np.mean(
        [
            all(task_preds[i].size == 1 for task_preds in predictions)
            for i in range(len(predictions[0]))
        ]
    )

def compute_taskwise_informativeness(predictions: list[list[np.ndarray]]) -> np.ndarray:
    """
    Computes informativeness for each task individually.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing
                                              a list of prediction arrays per sample.

    Returns:
        np.ndarray: An array containing informativeness for each task.
    """
    return np.array([compute_informativeness(task_preds) for task_preds in predictions])


def compute_efficiency(predictions: list[np.ndarray] | list[list[np.ndarray]]) -> float:
    """
    Computes efficiency as the average size of the prediction sets.

    Args:
        predictions (List[np.ndarray]): A list of prediction arrays per sample for a single task.

    Returns:
        float: Average number of predicted labels per sample.
    """
    if isinstance(predictions[0], np.ndarray):
        return np.mean([len(pred) for pred in predictions])

    return np.mean(
        [
            np.prod([len(task_preds[i]) for task_preds in predictions])
            for i in range(len(predictions[0]))
        ]
    )


def compute_taskwise_efficiency(predictions: list[list[np.ndarray]]) -> np.ndarray:
    """
    Computes efficiency for each task individually.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of 
                                              prediction arrays per sample.

    Returns:
        np.ndarray: An array containing efficiency for each task.
    """
    return np.array([compute_efficiency(task_preds) for task_preds in predictions])


def compute_weighted_efficiency(
    predictions: list[list[np.ndarray]],
    task_weights: np.ndarray,
    normalize_weights: bool = True,
) -> float:
    """
    Computes weighted efficiency across tasks based on provided weights.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list 
                                              of prediction arrays per sample.
        task_weights (np.ndarray): Weights for each task. Should have same length 
                                   as number of tasks.
        normalize_weights (bool): If True, weights will be normalized to sum to 1.

    Returns:
        float: Weighted average efficiency across tasks.
    """
    taskwise_eff = compute_taskwise_efficiency(predictions)

    if normalize_weights:
        task_weights = task_weights / np.sum(task_weights)

    return np.sum(taskwise_eff * task_weights)


def compute_weighted_informativeness(
    predictions: list[list[np.ndarray]],
    task_weights: np.ndarray,
    normalize_weights: bool = True,
) -> float:
    """
    Computes weighted informativeness across tasks based on provided weights.

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing a list of 
                                              prediction arrays per sample.
        task_weights (np.ndarray): Weights for each task. Should have same length 
                                   as number of tasks.
        normalize_weights (bool): If True, weights will be normalized to sum to 1.

    Returns:
        float: Weighted average informativeness across tasks.
    """
    taskwise_info = compute_taskwise_informativeness(predictions)

    if normalize_weights:
        task_weights = task_weights / np.sum(task_weights)

    return np.sum(taskwise_info * task_weights)


def compute_class_based_weights(task_num_classes: list[int]) -> np.ndarray:
    """
    Computes task weights based on the number of classes in each task.
    Tasks with fewer classes get lower weights.

    Args:
        task_num_classes (List[int]): List of number of classes for each task.

    Returns:
        np.ndarray: Weights for each task, normalized to sum to 1.
    """
    weights = np.array(task_num_classes)
    return weights / np.sum(weights)


def compute_classwise_coverage(
    predictions: list[np.ndarray], labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Computes empirical class-conditional coverage for a single task.

    Args:
        predictions (List[np.ndarray]): Prediction sets for each sample.
        labels (np.ndarray): True labels for each sample.
        num_classes (int): Number of classes in the task.

    Returns:
        np.ndarray: Classwise coverage rates (ĉ_y for each class y).
    """
    coverages = []
    for y in range(num_classes):
        idxs = np.where(labels == y)[0]
        if len(idxs) == 0:
            continue  # skip classes not present in the validation set
        cov_y = np.mean([labels[i] in predictions[i] for i in idxs])
        coverages.append(cov_y)
    return np.array(coverages)


def compute_taskwise_covgap(
    predictions: list[list[np.ndarray]],
    labels: list[np.ndarray],
    task_num_classes: list[int],
    alpha: float,
) -> list[float]:
    """
    Computes classes coverage gaps (CovGap).

    Args:
        predictions (List[List[np.ndarray]]): List of tasks, each containing prediction 
                                              sets per sample.
        labels (List[np.ndarray]): List of true label arrays for each task.
        task_num_classes (List[int]): Number of classes for each task.
        alpha (float): Significance level.

    Returns:
        float: CovGap values.
    """
    n_tasks = len(predictions)
    n_samples = len(labels[0])
    num_joint_classes = int(np.prod(task_num_classes))

    joint_labels = np.ravel_multi_index(
        tuple(labels[t] for t in range(n_tasks)),
        dims=tuple(task_num_classes),
    )

    covered = np.array([
        all(labels[t][i] in predictions[t][i] for t in range(n_tasks))
        for i in range(n_samples)
    ])

    class_gaps = []

    for y in range(num_joint_classes):
        idxs = np.where(joint_labels == y)[0]

        if len(idxs) == 0:
            continue

        class_coverage = np.mean(covered[idxs])
        class_gaps.append(abs(class_coverage - (1 - alpha)))

    return 100 * float(np.mean(class_gaps))


def compute_covgap(
    predictions: list[np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    alpha: float,
) -> float:
    """
    Compute average class coverage gap (CovGap) for a single task.
    """
    class_cov = compute_classwise_coverage(predictions, labels, num_classes)
    return 100 * np.mean(np.abs(class_cov - (1 - alpha)))
