from __future__ import annotations
import copy
import os
import yaml
import pandas as pd
import numpy as np

from typing import Any
import numpy as np

from core.predictor import ConformalPredictor
from data.datamodule import MultiOutputDataModule
from core.calibrators import HighLevelCalibrator, LowLevelCalibrator
from core.models import HighLevelModel, LowLevelModel

from src.metrics import (
    compute_taskwise_covgap,
    compute_overall_efficiency,
    compute_overall_informativeness,
    compute_taskwise_efficiency,
    compute_taskwise_informativeness,
    compute_efficiency,
    compute_informativeness,
    compute_covgap,
)

def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _validate_config(exp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[int], str]:
    data_cfg = exp.get("data_module", {})
    pred_cfg = exp.get("predictor", {})

    task_num_classes = data_cfg.get("task_num_classes")
    if not task_num_classes:
        raise ValueError("'task_num_classes' must be provided and non-empty.")

    cal_level = "high" if pred_cfg.get("calibrator_cls") == "HighLevelCalibrator" else "low"

    return data_cfg, pred_cfg, list(task_num_classes), cal_level


def _build_data_module(data_cfg: dict[str, Any], task_num_classes: list[int], iteration: int):
    return MultiOutputDataModule(
        root_dir=data_cfg["data_root"],
        task_num_classes=task_num_classes,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        split_idx=iteration,
    )


def _build_predictor(pred_cfg: dict[str, Any], task_num_classes: list[int]):
    return ConformalPredictor.load(
        model_cls=globals()[pred_cfg["model_cls"]],
        calibrator_cls=globals()[pred_cfg["calibrator_cls"]],
        task_num_classes=task_num_classes,
        alpha=pred_cfg["alpha"],
        artifacts_dir=pred_cfg["artifact_root"],
        nonconformity_key=pred_cfg["nonconformity_key"],
        cp_type=pred_cfg["cp_type"],
    )


def _extract_taskwise_true_labels(gt_labels: np.ndarray) -> list[np.ndarray]:
    gt_labels = np.asarray(gt_labels)
    if gt_labels.ndim != 2:
        raise ValueError(f"Expected gt_labels shape (n_samples, n_tasks), got {gt_labels.shape}")
    return [gt_labels[:, i] for i in range(gt_labels.shape[1])]


def _compute_high_level_metrics(
    prediction,
    y_trues: list[np.ndarray],
    task_num_classes: list[int],
    alpha: float,
) -> dict[str, Any]:
    return {
        "overall_efficiency": compute_overall_efficiency(prediction),
        "overall_informativeness": compute_overall_informativeness(prediction),
        "taskwise_efficiency": np.asarray(compute_taskwise_efficiency(prediction)),
        "taskwise_informativeness": np.asarray(compute_taskwise_informativeness(prediction)),
        "taskwise_covgap": np.asarray(
            compute_taskwise_covgap(prediction, y_trues, task_num_classes, alpha)
        ),
    }


def _compute_low_level_metrics(
    prediction,
    y_trues,
    task_num_classes,
    alpha: float,
) -> dict[str, Any]:
    return {
        "efficiency": compute_efficiency(prediction),
        "informativeness": compute_informativeness(prediction),
        "covgap": compute_covgap(prediction, y_trues, int(np.prod(task_num_classes)), alpha),
    }

def _summarize_high_level(
    experience_name: str,
    all_metrics: list[dict[str, Any]],
    n_tasks: int,
    task_num_classes: list[int],
) -> dict[str, float]:
    overall_eff = np.array([m["overall_efficiency"] for m in all_metrics])
    overall_info = np.array([m["overall_informativeness"] for m in all_metrics])
    taskwise_eff = np.array([m["taskwise_efficiency"] for m in all_metrics])         # (n_runs, n_tasks)
    taskwise_info = np.array([m["taskwise_informativeness"] for m in all_metrics])   # (n_runs, n_tasks)
    taskwise_covgap = np.array([m["taskwise_covgap"] for m in all_metrics])          # (n_runs, n_tasks)

    class_weights = np.array(task_num_classes) / np.sum(task_num_classes)
    overall_covgap = np.sum(taskwise_covgap * class_weights, axis=1)

    results = {
        "experience_name": experience_name,
        "overall_efficiency_mean": float(np.mean(overall_eff)),
        "overall_efficiency_std": float(np.std(overall_eff)),
        "overall_informativeness_mean": float(np.mean(overall_info)),
        "overall_informativeness_std": float(np.std(overall_info)),
        "overall_covgap_mean": float(np.mean(overall_covgap)),
        "overall_covgap_std": float(np.std(overall_covgap)),
    }

    for i in range(n_tasks):
        results[f"taskwise_efficiency_mean_{i}"] = float(np.mean(taskwise_eff[:, i]))
        results[f"taskwise_efficiency_std_{i}"] = float(np.std(taskwise_eff[:, i]))
        results[f"taskwise_informativeness_mean_{i}"] = float(np.mean(taskwise_info[:, i]))
        results[f"taskwise_informativeness_std_{i}"] = float(np.std(taskwise_info[:, i]))
        results[f"taskwise_covgap_mean_{i}"] = float(np.mean(taskwise_covgap[:, i]))
        results[f"taskwise_covgap_std_{i}"] = float(np.std(taskwise_covgap[:, i]))

    return results


def _summarize_low_level(experience_name: str, all_metrics: list[dict[str, Any]]) -> dict[str, float]:
    efficiency = np.array([m["efficiency"] for m in all_metrics])
    informativeness = np.array([m["informativeness"] for m in all_metrics])
    covgap = np.array([m["covgap"] for m in all_metrics])

    return {
        "experience_name": experience_name,
        "efficiency_mean": float(np.mean(efficiency)),
        "efficiency_std": float(np.std(efficiency)),
        "informativeness_mean": float(np.mean(informativeness)),
        "informativeness_std": float(np.std(informativeness)),
        "covgap_mean": float(np.mean(covgap)),
        "covgap_std": float(np.std(covgap)),
    }


def run(exp: dict[str, Any]) -> dict[str, float]:
    print(f"Running experiment: {exp['name']}")

    data_cfg, pred_cfg, task_num_classes, cal_level = _validate_config(exp)
    n_runs = data_cfg.get("max_iter", 1)
    n_tasks = len(task_num_classes)

    all_metrics: list[dict[str, Any]] = []

    for iteration in range(n_runs):
        dm = _build_data_module(data_cfg, task_num_classes, iteration)
        predictor = _build_predictor(pred_cfg, task_num_classes)

        predictor.fit(
            data_module=dm,
            train_model=pred_cfg["train_model"],
            calibrate_model=pred_cfg["calibrate_model"],
            alpha=pred_cfg["alpha"],
        )

        prediction = predictor.predict(dm.test_dataloader())
        gt_labels = predictor.get_labels(dm.datasets["test"])

        if cal_level == "high":
            y_trues = _extract_taskwise_true_labels(gt_labels)
            metrics = _compute_high_level_metrics(
                prediction=prediction,
                y_trues=y_trues,
                task_num_classes=task_num_classes,
                alpha=pred_cfg["alpha"],
            )

        else:  # cal_level == "low"
            metrics = _compute_low_level_metrics(
                prediction=prediction,
                y_trues=gt_labels,
                task_num_classes=task_num_classes,
                alpha=pred_cfg["alpha"],
            )

        all_metrics.append(metrics)

    if cal_level == "high":
        return _summarize_high_level(exp["name"], all_metrics, n_tasks, task_num_classes)

    return _summarize_low_level(exp["name"], all_metrics)

def run_experiments(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config["experiments"]

    resolved_experiments = [
        deep_merge(defaults, exp)
        for exp in experiments
    ]

    name = config_path.split("/")[-1].replace(".yaml", "")
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([run(exp) for exp in resolved_experiments]).to_csv(f"./results/{name}-results.csv", index=False)