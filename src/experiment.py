from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.datamodule import MultiOutputDataModule
from core.calibrators import LowLevelCalibrator, HighLevelCalibrator
from core.models import LowLevelModel, HighLevelModel
from core.predictor import ConformalPredictor
from src.metrics import (
    compute_covgap,
    compute_efficiency,
    compute_informativeness,
    compute_taskwise_covgap,
    compute_taskwise_efficiency,
    compute_taskwise_informativeness,
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


def _build_data_module(
    data_cfg: dict[str, Any], task_num_classes: list[int], iteration: int
) -> MultiOutputDataModule:
    return MultiOutputDataModule(
        root_dir=data_cfg["data_root"],
        task_num_classes=task_num_classes,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        split_idx=iteration,
    )


def _build_predictor(pred_cfg: dict[str, Any], task_num_classes: list[int]) -> ConformalPredictor:
    return ConformalPredictor.load(
        model_cls=globals()[pred_cfg["model_cls"]],
        calibrator_cls=globals()[pred_cfg["calibrator_cls"]],
        task_num_classes=task_num_classes,
        alpha=pred_cfg["alpha"],
        artifacts_dir=pred_cfg["artifact_root"],
        nonconformity_key=pred_cfg["nonconformity_key"],
        cp_type=pred_cfg["cp_type"],
    )


def _compute_high_level_metrics(
    prediction,
    y_trues: list[np.ndarray],
    task_num_classes: list[int],
    alpha: float,
) -> dict[str, Any]:
    return {
        "efficiency": compute_efficiency(prediction),
        "informativeness": compute_informativeness(prediction),
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
) -> dict[tuple[str, str], float | str]:
    overall_eff = np.array([m["efficiency"] for m in all_metrics])
    overall_info = np.array([m["informativeness"] for m in all_metrics])

    taskwise_eff = np.array([m["taskwise_efficiency"] for m in all_metrics])
    taskwise_info = np.array([m["taskwise_informativeness"] for m in all_metrics])
    taskwise_covgap = np.array([m["taskwise_covgap"] for m in all_metrics])

    class_weights = np.array(task_num_classes) / np.sum(task_num_classes)
    overall_covgap = np.sum(taskwise_covgap * class_weights, axis=1)

    results = {
        ("Experience", ""): experience_name,

        ("Overall Eff", "mean"): float(np.mean(overall_eff)),
        ("Overall Eff", "std"): float(np.std(overall_eff)),

        ("Overall Inf", "mean"): float(np.mean(overall_info)),
        ("Overall Inf", "std"): float(np.std(overall_info)),

        ("Overall CovGap", "mean"): float(np.mean(overall_covgap)),
        ("Overall CovGap", "std"): float(np.std(overall_covgap)),
    }

    for i in range(n_tasks):
        results[(f"{i} - Task Eff", "mean")] = float(np.mean(taskwise_eff[:, i]))
        results[(f"{i} - Task Eff", "std")] = float(np.std(taskwise_eff[:, i]))

        results[(f"{i} - Task Inf", "mean")] = float(np.mean(taskwise_info[:, i]))
        results[(f"{i} - Task Inf", "std")] = float(np.std(taskwise_info[:, i]))

        results[(f"{i} - Task CovGap", "mean")] = float(np.mean(taskwise_covgap[:, i]))
        results[(f"{i} - Task CovGap", "std")] = float(np.std(taskwise_covgap[:, i]))

    return results


def _summarize_low_level(
    experience_name: str,
    all_metrics: list[dict[str, Any]],
) -> dict[tuple[str, str], float | str]:
    efficiency = np.array([m["efficiency"] for m in all_metrics])
    informativeness = np.array([m["informativeness"] for m in all_metrics])
    covgap = np.array([m["covgap"] for m in all_metrics])

    return {
        ("Experience", ""): experience_name,

        ("Overall Eff", "mean"): float(np.mean(efficiency)),
        ("Overall Eff", "std"): float(np.std(efficiency)),

        ("Overall Inf", "mean"): float(np.mean(informativeness)),
        ("Overall Inf", "std"): float(np.std(informativeness)),

        ("Overall CovGap", "mean"): float(np.mean(covgap)),
        ("Overall CovGap", "std"): float(np.std(covgap)),
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
            metrics = _compute_high_level_metrics(
                prediction=prediction,
                y_trues=gt_labels,
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

def run_experiments(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config["experiments"]

    resolved_experiments = [
        deep_merge(defaults, exp)
        for exp in experiments
    ]

    name = config_path.split("/")[-1].replace(".yaml", "")
    os.makedirs("./results", exist_ok=True)
    results_df = pd.DataFrame([run(exp) for exp in resolved_experiments])
    results_df.to_csv(
        f"./results/{name}-results.csv",
        index=False,
        float_format="%.4f",
    )