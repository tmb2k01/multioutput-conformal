import json
import math
import os
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch

import wandb
from src.calibration.calibration import calibration
from src.data.multi_output_dataset import MultiOutputDataModule
from src.models.high_level_model import HighLevelModel
from src.models.low_level_model import LowLevelModel
from src.models.model_utils import convert_multitask_preds


def convert_numpy_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(x) for x in obj]
    else:
        return obj


def calibrate_model(
    model: Union[HighLevelModel, LowLevelModel],
    datamodule: MultiOutputDataModule,
    filename: str,
    alpha: float = 0.05,
    n_clusters: Union[str, int] = "auto",
):
    high_level = isinstance(model, HighLevelModel)

    model.eval()
    trainer = pl.Trainer()
    preds = trainer.predict(model, dataloaders=datamodule.calib_dataloader())

    if high_level:
        calib_preds = convert_multitask_preds(preds)
    else:
        calib_preds = np.concatenate(preds, axis=0)
        calib_preds = np.array(calib_preds)

    true_labels = np.stack(
        [labels for _, labels in datamodule.datasets["calib"]], axis=1
    )

    if not high_level:
        task_classes = datamodule.task_num_classes
        multiplier = np.array(
            [math.prod(task_classes[i + 1 :]) for i in range(len(task_classes))]
        )
        true_labels = (true_labels * multiplier[:, None]).sum(axis=0)

    q_hats = calibration(
        calib_preds,
        true_labels,
        high_level=high_level,
        alpha=alpha,
        n_clusters=n_clusters,
    )

    with open(f"./models/{filename}-calibration.json", "w") as f:
        json.dump(convert_numpy_to_native(q_hats), f, indent=2)


def init_wandb_logger(project_name: str):
    key_file = os.environ.get("WANDB_API_KEY_FILE", "./.wandb_api_key")
    key_path = Path(key_file)

    if not key_path.is_file():
        print(
            f"W&B key file '{key_path}' not found. Continuing without Weights & Biases logging."
        )
        return None

    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        print(
            f"W&B key file '{key_path}' is empty. Continuing without Weights & Biases logging."
        )
        return None

    os.environ.setdefault("WANDB_API_KEY", key)

    try:
        wandb.login(key=key, relogin=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to authenticate with Weights & Biases: {exc}")
        print("Continuing without W&B logging.")
        return None

    try:
        return pl.loggers.WandbLogger(project=project_name)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to initialise W&B logger: {exc}")
        print("Continuing without W&B logging.")
        return None


def train_model(
    root_dir,
    filename,
    task_num_classes,
    model: Union[HighLevelModel, LowLevelModel],
    model_dir: str = "models/",
    alpha: float = 0.05,
    calibration_clusters: Union[str, int] = "auto",
):
    datamodule = MultiOutputDataModule(
        root_dir=root_dir,
        task_num_classes=task_num_classes,
        batch_size=64,
        num_workers=8,
    )
    datamodule.setup()
    model = model(task_num_classes=task_num_classes)
    wandb_logger = init_wandb_logger(project_name=f"{filename}-model")
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_dir,
        filename=f"{filename}-model",
        save_top_k=1,
        save_weights_only=False,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=[early_stopping, checkpoint],
        logger=wandb_logger if wandb_logger is not None else True,
    )
    trainer.fit(model, datamodule)
    if wandb_logger is not None:
        wandb.finish()

    # Calibration logic goes here
    model_path = checkpoint.best_model_path
    model = model.__class__.load_from_checkpoint(
        model_path, task_num_classes=task_num_classes
    )
    calibrate_model(
        model,
        datamodule,
        filename=filename,
        alpha=alpha,
        n_clusters=calibration_clusters,
    )


CONFIG_PATH = Path("./static/train-config.json")


def train():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    os.makedirs("models", exist_ok=True)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    alpha = cfg.get("alpha", 0.05)
    calibration_clusters = cfg.get("calibration_clusters", "auto")
    model_dir = cfg.get("models_dir", "models/")

    for ds in cfg.get("datasets", []):
        name = ds["name"]
        root_dir = ds["root_dir"]
        task_num_classes = ds["task_num_classes"]
        levels = ds.get("levels", ["HIGH", "LOW"])

        for level in levels:
            is_high = level.upper() == "HIGH"
            model_cls = HighLevelModel if is_high else LowLevelModel
            filename = f"{name.lower()}-{'high' if is_high else 'low'}-level"

            train_model(
                root_dir=root_dir,
                filename=filename,
                task_num_classes=task_num_classes,
                model=model_cls,
                model_dir=model_dir,
                alpha=alpha,
                calibration_clusters=calibration_clusters,
            )
