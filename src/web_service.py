import gradio as gr
from PIL import Image

CONFIG = {
    "SGVehicle": {
        "tasks": [
            {"name": "Color", "classes": ["Blue", "Red", "Yellow"]},
            {"name": "Type", "classes": ["Car", "Motorcycle", "Truck"]},
        ]
    },
    "UKTFaces": {
        "tasks": [
            {"name": "Gender", "classes": ["Male", "Female"]},
            {"name": "Expression", "classes": ["Happy", "Sad", "Neutral"]},
        ]
    },
}

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


def generate_table_data(model_type):
    """Generate rows: each row = [Task, Class, Score, Threshold]"""
    if model_type not in CONFIG:
        return []

    rows = []
    for task in CONFIG[model_type]["tasks"]:
        rows.append([f"🔷 {task['name']}", "", "", ""])  # header row for task
        for cls in task["classes"]:
            rows.append(["", f"{'🟢 ' if True else ""} {cls}", "-", "-"])
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

        def update_table(model_type):
            return generate_table_data(model_type)

        model_level.change(fn=update_cp_type, inputs=model_level, outputs=cp_type)
        model_type.change(fn=update_table, inputs=model_type, outputs=table_output)

    interface.launch()
