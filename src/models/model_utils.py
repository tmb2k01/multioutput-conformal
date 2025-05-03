import numpy as np


def convert_multitask_preds(calib_preds):
    """
    Convert multi-task model predictions into per-task NumPy arrays.

    Args:
        calib_preds (List[List[np.ndarray]]): Output from `trainer.predict(model, dataloader)`.
            Each element is a batch, and contains a list of arrays (one per task).

    Returns:
        List[np.ndarray]: One NumPy array per task, with shape (N, C_task_i), where N is the total
                          number of samples across all batches, and C_task_i is the number of classes
                          for task i.
    """
    if not calib_preds or not isinstance(calib_preds[0], list):
        raise ValueError(
            "Expected `calib_preds` to be a list of lists (batches of task outputs)"
        )

    T = len(calib_preds[0])  # number of tasks
    task_outputs = [[] for _ in range(T)]

    for batch in calib_preds:
        if len(batch) != T:
            raise ValueError("Inconsistent number of tasks across batches")
        for t, task_output in enumerate(batch):
            task_outputs[t].append(task_output)

    calib_preds_np = [np.concatenate(task, axis=0) for task in task_outputs]
    return calib_preds_np
