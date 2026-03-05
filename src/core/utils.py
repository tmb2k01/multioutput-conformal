import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def expand_path(p: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p))))

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def convert_numpy_to_native(obj: Any) -> Any:
    """Make calibration output JSON serializable."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_native(x) for x in obj]
    return obj

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

    return [np.concatenate([_to_numpy(b) for b in task], axis=0) for task in task_outputs]
