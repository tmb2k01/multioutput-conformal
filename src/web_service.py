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
CONFIG_PATH = Path("./static/config.json")
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


def generate_table_data(
    model_type: str, level: str = "HIGH", scores=None
) -> list[list[str]]:
    """Generate prediction table rows with optional score values."""
    if model_type not in CONFIG:
        return []

    rows = []
    tasks = CONFIG[model_type]["tasks"]

    if level == "HIGH":
        idx = 0
        for task in tasks:
            rows.append([f"🔷 {task['name']}", "", "", ""])
            for cls in task["classes"]:
                score = f"{scores[idx]:.3f}" if scores else "-"
                rows.append(["", cls, score, "-"])
                idx += 1
    else:
        combined_classes = list(product(*[t["classes"] for t in tasks]))
        for i, combo in enumerate(combined_classes):
            label = " | ".join(combo)
            score = f"{scores[i]:.3f}" if scores else "-"
            rows.append([label, score, "-"])

    return rows


def predict(image, level, model_type, nonconformity_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(level, model_type).to(device)

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

    table = generate_table_data(model_type, level, scores=scores.tolist())
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

        # --- Logic ---
        model_level.change(
            lambda lvl: gr.update(
                choices=LOW_CP_TYPES if lvl == "LOW" else HIGH_CP_TYPES,
                value=(LOW_CP_TYPES if lvl == "LOW" else HIGH_CP_TYPES)[0],
            ),
            inputs=model_level,
            outputs=cp_type,
        )

        def update_table(level, model_type):
            new_headers = (
                ["Class", "Score", "Threshold"] if level == "LOW" else column_names
            )
            new_rows = generate_table_data(model_type, level)
            return gr.update(headers=new_headers, value=new_rows)

        model_level.change(update_table, [model_level, model_type], table_output)
        model_type.change(update_table, [model_level, model_type], table_output)

        submit_btn.click(
            predict, [image_input, model_level, model_type, nonconformity], table_output
        )

        image_input.change(
            lambda img: gr.update(interactive=img is not None),
            inputs=image_input,
            outputs=submit_btn,
        )

    ui.launch()
