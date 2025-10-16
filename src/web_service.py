import json
from itertools import product
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T

from src.calibration.nonconformity_functions import NONCONFORMITY_FN_DIC
from src.models.high_level_model import HighLevelModel
from src.models.low_level_model import LowLevelModel

# Constants & Config
CONFIG_PATH = Path("./static/ws-config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

LOW_CP_TYPES = [
    "Standard CP - Global Threshold",
    "Classwise CP",
    "Clustered CP - Global Clusters",
]
HIGH_CP_TYPES = [
    "Standard CP - Global Threshold",
    "Standard CP - Taskwise Threshold",
    "Classwise CP",
    "Clustered CP - Global Clusters",
    "Clustered CP - Taskwise Clusters",
]

CP_TYPE_MAPPING = {
    "Standard CP - Global Threshold": "scp_global",
    "Standard CP - Taskwise Threshold": "scp_task",
    "Classwise CP": "ccp_class",
    "Clustered CP - Global Clusters": "ccp_global_cluster",
    "Clustered CP - Taskwise Clusters": "ccp_task_cluster",
}

model_cache = {}


def get_task_num_classes(model_type: str) -> list[int]:
    return [len(task["classes"]) for task in CONFIG[model_type]["tasks"]]


def load_model(level: str, model_type: str):
    key = f"{level}_{model_type}"
    if key in model_cache:
        return model_cache[key]

    num_classes = get_task_num_classes(model_type)
    ckpt = CONFIG[model_type]["weights"][level.lower()]
    model_cls = HighLevelModel if level == "HIGH" else LowLevelModel

    model = model_cls.load_from_checkpoint(ckpt, task_num_classes=num_classes)
    model.eval()
    model_cache[key] = model
    return model


# ---- New helper functions to clean up generate_table_data ----


def load_thresholds(model_type: str, level: str, score_type: str):
    threshold_path = CONFIG[model_type]["thresholds"][level.lower()]
    with open(threshold_path, "r") as f:
        all_thresholds = json.load(f)
    return all_thresholds.get(score_type, {})


def get_threshold_for_high_level(
    cp_type: str, thresholds_data: dict, task_idx: int, class_idx: int, idx: int
) -> str:
    if cp_type == "scp_global":
        return f"{thresholds_data['scp_global_threshold']:.5f}"
    if cp_type == "scp_task":
        return f"{thresholds_data['scp_task_thresholds'][task_idx]:.5f}"
    if cp_type == "ccp_class":
        # Use class_idx here instead of idx
        return f"{thresholds_data['ccp_class_thresholds'][task_idx][class_idx]:.5f}"
    if cp_type == "ccp_task_cluster":
        cluster_id = thresholds_data["ccp_task_cluster_thresholds"][f"task-{task_idx}"][
            "mapping"
        ][class_idx]
        qhat = thresholds_data["ccp_task_cluster_thresholds"][f"task-{task_idx}"][
            "qhats"
        ][cluster_id]
        return f"{qhat:.5f}"
    if cp_type == "ccp_global_cluster":
        cluster_id = thresholds_data["ccp_global_cluster_thresholds"][
            "class_to_cluster_mapping"
        ][f"task-{task_idx}"][class_idx]
        qhat = thresholds_data["ccp_global_cluster_thresholds"]["cluster_qhats"][
            cluster_id
        ]
        return f"{qhat:.5f}"
    return "-"


def get_threshold_for_low_level(cp_type: str, thresholds_data: dict, idx: int) -> str:
    if cp_type == "scp_global":
        return f"{thresholds_data['scp_global_threshold']:.5f}"
    if cp_type == "ccp_class":
        return f"{thresholds_data['ccp_class_thresholds'][idx]:.5f}"
    if cp_type == "ccp_global_cluster":
        cluster_id = thresholds_data["ccp_global_clusters"]["class_to_cluster_mapping"][
            idx
        ]
        qhat = thresholds_data["ccp_global_clusters"]["cluster_qhats"][cluster_id]
        return f"{qhat:.5f}"
    return "-"


def score_passed(score: str, threshold: str) -> bool:
    if score == "-" or threshold == "-":
        return False
    return float(score) <= float(threshold)


def generate_table_data(
    model_type: str,
    level: str = "HIGH",
    scores=None,
    cp_type: str = "scp_global",
    score_type: str = "hinge",
) -> list[list[str]]:
    """Generate prediction table rows with optional score and threshold values."""
    if model_type not in CONFIG:
        return []

    thresholds_data = load_thresholds(model_type, level, score_type)
    tasks = CONFIG[model_type]["tasks"]

    rows = []

    if level == "HIGH":
        idx = 0
        for task_idx, task in enumerate(tasks):
            rows.append([f"🔷 {task['name']}", "", "", ""])
            for class_idx, cls in enumerate(task["classes"]):
                score = f"{scores[idx]:.5f}" if scores is not None else "-"
                threshold = get_threshold_for_high_level(
                    cp_type, thresholds_data, task_idx, class_idx, idx
                )
                mark = "🟢" if score_passed(score, threshold) else ""
                rows.append(["", f"{mark}{cls}", score, threshold])
                idx += 1

    else:  # LOW level
        combined_classes = list(product(*[t["classes"] for t in tasks]))
        for i, combo in enumerate(combined_classes):
            label = " | ".join(combo)
            score = f"{scores[i]:.5f}" if scores is not None else "-"
            threshold = get_threshold_for_low_level(cp_type, thresholds_data, i)
            mark = "🟢" if score_passed(score, threshold) else ""
            rows.append([f"{mark}{label}", score, threshold])

    return rows


def predict(image, level, model_type, nonconformity_type, cp_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(level, model_type).to(device)

    nonconformity_type = nonconformity_type.lower()
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    if isinstance(outputs, (list, tuple)):
        softmax_outputs = [out.softmax(dim=-1).cpu().numpy() for out in outputs]
    else:
        softmax_outputs = outputs.softmax(dim=-1).cpu().numpy()

    nonconformity_fn = NONCONFORMITY_FN_DIC[nonconformity_type]
    scores = nonconformity_fn(softmax_outputs)

    if isinstance(scores, list):
        scores = np.concatenate(scores, axis=1).flatten()
    else:
        scores = scores.flatten()

    internal_cp_type = CP_TYPE_MAPPING.get(cp_type, "scp_global")
    table = generate_table_data(
        model_type,
        level,
        scores=scores.tolist(),
        cp_type=internal_cp_type,
        score_type=nonconformity_type,
    )
    return table


def launch():
    default_model = next(iter(CONFIG))
    column_names = ["Task", "Class", "Score", "Threshold"]

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
            headers=column_names,
            value=generate_table_data(default_model),
            interactive=False,
            wrap=True,
            elem_classes="no-select",
        )

        def update_cp_choices(lvl):
            choices = LOW_CP_TYPES if lvl == "LOW" else HIGH_CP_TYPES
            return gr.update(choices=choices, value=choices[0])

        model_level.change(update_cp_choices, model_level, cp_type)

        def update_table(level, model_type, cp_type, nonconformity):
            internal_cp_type = CP_TYPE_MAPPING.get(cp_type, "scp_global")
            score_type = nonconformity.lower()
            new_headers = (
                ["Class", "Score", "Threshold"] if level == "LOW" else column_names
            )
            new_rows = generate_table_data(
                model_type, level, cp_type=internal_cp_type, score_type=score_type
            )
            return gr.update(headers=new_headers, value=new_rows)

        for widget in [model_level, model_type, cp_type, nonconformity]:
            widget.change(
                update_table,
                [model_level, model_type, cp_type, nonconformity],
                table_output,
            )

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
