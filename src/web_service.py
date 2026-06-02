import json
import os
from itertools import product
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor

from core.calibrators import (
    BaseCalibrator,
    HighLevelCalibrator,
    LowLevelCalibrator,
)
from core.models import BaseModel, HighLevelModel, LowLevelModel
from core.predictor import ConformalPredictor
from core.utils import expand_path

# ---------- config ----------
DEFAULT_CONFIG_PATH = Path("./static/ws-config.json")
CONFIG = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts/artifacts")

# Friendly CP names -> calibration_fn_key used by the calibrators / threshold files.
CP_TYPE_MAPPING = {
    "Standard CP - Global Threshold": "scp_global_threshold",
    "Standard CP - Taskwise Threshold": "scp_task_thresholds",
    "Classwise CP": "ccp_class_thresholds",
    "Clustered CP - Global Clusters": "ccp_global_cluster_thresholds",
    "Clustered CP - Taskwise Clusters": "ccp_task_cluster_thresholds",
}
LOW_CP_TYPES = [
    "Standard CP - Global Threshold",
    "Classwise CP",
    "Clustered CP - Global Clusters",
]
HIGH_CP_TYPES = list(CP_TYPE_MAPPING.keys())

# Image preprocessing mirrors data.datamodule (to_tensor + Resize((256, 256))).
_RESIZE = Resize((256, 256))

# Cache loaded predictors keyed by configuration.
_predictor_cache: dict[tuple[str, str, str, str, float], ConformalPredictor] = {}


def get_task_num_classes(model_type: str) -> list[int]:
    return [len(task["classes"]) for task in CONFIG[model_type]["tasks"]]


class _SingleImageDataset(Dataset):
    """Wraps a single preprocessed image so it can flow through the predictor pipeline."""

    def __init__(self, image: Image.Image, n_tasks: int) -> None:
        self.image = _RESIZE(to_tensor(image.convert("RGB")))
        self.n_tasks = n_tasks

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image, torch.zeros(self.n_tasks, dtype=torch.long)


def _build_predictor(
    model_type: str,
    level: str,
    nonconformity_key: str,
    cp_type: str,
    alpha: float,
) -> ConformalPredictor:
    cache_key = (model_type, level, nonconformity_key, cp_type, alpha)
    if cache_key in _predictor_cache:
        return _predictor_cache[cache_key]

    is_high = level == "HIGH"
    artifacts_dir = expand_path(ARTIFACTS_ROOT) / CONFIG[model_type]["artifacts"][level.lower()]

    model_cls: type[BaseModel] = HighLevelModel if is_high else LowLevelModel
    calibrator_cls: type[BaseCalibrator] = (
        HighLevelCalibrator if is_high else LowLevelCalibrator
    )

    predictor = ConformalPredictor.load(
        model_cls=model_cls,
        calibrator_cls=calibrator_cls,
        task_num_classes=get_task_num_classes(model_type),
        alpha=alpha,
        artifacts_dir=artifacts_dir,
        nonconformity_key=nonconformity_key,
        cp_type=cp_type,
    )
    _predictor_cache[cache_key] = predictor
    return predictor


def _included_per_task(
    prediction: list[np.ndarray] | list[list[np.ndarray]], level: str
) -> list[set[int]]:
    """Reduce a single-sample prediction into per-task sets of included class indices."""
    if level == "HIGH":
        return [set(np.asarray(task_pred[0]).tolist()) for task_pred in prediction]
    return [set(np.asarray(prediction[0]).tolist())]


def _build_table(
    model_type: str,
    level: str,
    prediction: list[np.ndarray] | list[list[np.ndarray]] | None = None,
) -> list[list[str]]:
    tasks = CONFIG[model_type]["tasks"]
    included = _included_per_task(prediction, level) if prediction is not None else None
    rows: list[list[str]] = []

    if level == "HIGH":
        for task_idx, task in enumerate(tasks):
            in_set = included[task_idx] if included is not None else set()
            rows.append([f"🔷 {task['name']}", ""])
            for class_idx, cls in enumerate(task["classes"]):
                mark = "🟢 " if class_idx in in_set else ""
                rows.append(["", f"{mark}{cls}"])
    else:  # LOW level: joint classes over the cartesian product of task classes.
        in_set = included[0] if included is not None else set()
        combined_classes = list(product(*[t["classes"] for t in tasks]))
        for i, combo in enumerate(combined_classes):
            mark = "🟢 " if i in in_set else ""
            rows.append([f"{mark}" + " | ".join(combo)])

    return rows


def predict(
    image: Image.Image | None,
    level: str,
    model_type: str,
    nonconformity: str,
    cp_type: str,
) -> list[list[str]]:
    if image is None:
        return _build_table(model_type, level)

    nonconformity_key = nonconformity.lower()
    internal_cp_type = CP_TYPE_MAPPING.get(cp_type, "scp_global_threshold")
    alpha = float(CONFIG[model_type].get("alpha", 0.05))

    try:
        predictor = _build_predictor(
            model_type, level, nonconformity_key, internal_cp_type, alpha
        )
        loader = DataLoader(
            _SingleImageDataset(image, len(get_task_num_classes(model_type))),
            batch_size=1,
        )
        prediction = predictor.predict(loader)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        return [[f"Unavailable for this configuration: {e}"]]

    return _build_table(model_type, level, prediction)


def launch(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    global CONFIG
    CONFIG = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _predictor_cache.clear()

    default_model = next(iter(CONFIG))
    high_columns = ["Task", "Class"]

    with gr.Blocks(css_paths="./static/styles.css") as ui:
        gr.Markdown("# Multi-output Conformal Prediction")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    sources=["upload", "clipboard"],
                    height=292,
                )
                submit_btn = gr.Button("Predict", variant="primary", interactive=False)

            with gr.Column():
                model_level = gr.Radio(
                    ["LOW", "HIGH"], value="HIGH", label="Model Level"
                )
                model_type = gr.Dropdown(
                    choices=list(CONFIG), value=default_model, label="Model Type"
                )
                nonconformity = gr.Dropdown(
                    ["Hinge", "Margin", "PiP"],
                    value="Hinge",
                    label="Nonconformity Score",
                )
                cp_type = gr.Dropdown(
                    HIGH_CP_TYPES, value=HIGH_CP_TYPES[0], label="Conf. Pred. Type"
                )

        gr.Markdown("# Prediction Results")
        table_output = gr.Dataframe(
            headers=high_columns,
            value=_build_table(default_model, "HIGH"),
            interactive=False,
            wrap=True,
            elem_classes="no-select",
        )

        def update_cp_choices(lvl: str) -> dict:
            choices = LOW_CP_TYPES if lvl == "LOW" else HIGH_CP_TYPES
            return gr.update(choices=choices, value=choices[0])

        model_level.change(update_cp_choices, model_level, cp_type)

        def update_table(level: str, model_type: str) -> dict:
            headers = ["Class"] if level == "LOW" else high_columns
            return gr.update(headers=headers, value=_build_table(model_type, level))

        for widget in [model_level, model_type]:
            widget.change(update_table, [model_level, model_type], table_output)

        submit_btn.click(
            predict,
            [image_input, model_level, model_type, nonconformity, cp_type],
            table_output,
        )

        image_input.change(
            lambda img: gr.update(interactive=img is not None),
            inputs=image_input,
            outputs=submit_btn,
        )

    ui.launch()
