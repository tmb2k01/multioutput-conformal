import json
from itertools import product

import gradio as gr
from PIL import Image

CONFIG_SRC = "./static/config.json"
CONFIG = json.load(open(CONFIG_SRC, "r", encoding="utf-8"))

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


def generate_table_data(model_type, level="HIGH"):
    """Generate rows for table based on model level."""
    if model_type not in CONFIG:
        return []

    rows = []

    if level == "HIGH":
        for task in CONFIG[model_type]["tasks"]:
            rows.append([f"🔷 {task['name']}", "", "", ""])
            for cls in task["classes"]:
                rows.append(["", f"{'🟢 ' if True else ''} {cls}", "-", "-"])

    else:  # LOW level
        all_class_lists = [task["classes"] for task in CONFIG[model_type]["tasks"]]
        combined_classes = list(product(*all_class_lists))

        for combo in combined_classes:
            class_label = " | ".join(combo)
            rows.append([f"{'🟢 ' if True else ''} {class_label}", "-", "-"])

    return rows


def launch():
    default_model = list(CONFIG.keys())[0]
    column_names = ["Task", "Class", "Score", "Threshold"]

    with gr.Blocks(css_paths="./static/styles.css") as interface:
        gr.Markdown("# Multi-output Conformal Prediction")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil", label="Upload Image", sources=["upload", "clipboard"]
                )
                submit_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                model_level = gr.Radio(
                    choices=["LOW", "HIGH"], value="HIGH", label="Model Level"
                )

                model_type = gr.Dropdown(
                    choices=list(CONFIG.keys()), value=default_model, label="Model Type"
                )

                nonconformity = gr.Dropdown(
                    choices=["Hinge", "Margin", "PiP"],
                    value="Hinge",
                    label="Nonconformity Score",
                    interactive=True,
                )

                cp_type = gr.Dropdown(
                    choices=HIGH_CP_TYPES,
                    value=HIGH_CP_TYPES[0],
                    label="Conf. Pred. Type",
                    interactive=True,
                )

        gr.Markdown("# Prediction Results")
        table_output = gr.Dataframe(
            headers=column_names,
            value=generate_table_data(default_model),
            interactive=False,
            wrap=True,
            elem_classes="no-select",
        )

        def update_cp_type(level):
            if level == "LOW":
                return gr.update(choices=LOW_CP_TYPES, value=LOW_CP_TYPES[0])
            else:
                return gr.update(
                    choices=HIGH_CP_TYPES,
                    value=HIGH_CP_TYPES[0],
                )

        def update_table_and_columns(level, model_type):
            if level == "LOW":
                new_columns = ["Class", "Score", "Threshold"]
            else:
                new_columns = ["Task", "Class", "Score", "Threshold"]

            new_data = generate_table_data(model_type, level)
            return gr.update(headers=new_columns, value=new_data)

        model_level.change(fn=update_cp_type, inputs=model_level, outputs=cp_type)

        model_level.change(
            fn=update_table_and_columns,
            inputs=[model_level, model_type],
            outputs=table_output,
        )
        model_type.change(
            fn=update_table_and_columns,
            inputs=[model_level, model_type],
            outputs=table_output,
        )

    interface.launch()
