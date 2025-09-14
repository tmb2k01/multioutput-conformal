import json
import math
import os
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

# SGVehicle Dataset properties
SGVEHICLE_COLOR = 12
SGVEHICLE_TYPE = 11
SGVEHICLE_TASK_NUM_CLASSES = [SGVEHICLE_COLOR, SGVEHICLE_TYPE]

UTKFACE_GENDER = 2
UTKFACE_RACE = 5
UTKFACE_TASK_NUM_CLASSES = [UTKFACE_GENDER, UTKFACE_RACE]


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
    trainer = pl.Trainer(accelerator="gpu")
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


def train_model(
    root_dir,
    filename,
    task_num_classes,
    model: Union[HighLevelModel, LowLevelModel],
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
    wandb_logger = pl.loggers.WandbLogger(
        project=f"{filename}-model",
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/",
        filename=f"{filename}-model",
        save_top_k=1,
        save_weights_only=False,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=30,
        callbacks=[early_stopping, checkpoint],
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule)
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


def train():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    wandb.login()
    os.makedirs("models", exist_ok=True)
    train_model(
        "data/SGVehicle",
        "sgvehicle-high-level",
        SGVEHICLE_TASK_NUM_CLASSES,
        HighLevelModel,
    )
    train_model(
        "data/SGVehicle",
        "sgvehicle-low-level",
        SGVEHICLE_TASK_NUM_CLASSES,
        LowLevelModel,
    )
    train_model(
        "data/UTKFace",
        "utkface-high-level",
        UTKFACE_TASK_NUM_CLASSES,
        HighLevelModel,
    )
    train_model(
        "data/UTKFace", "utkface-low-level", UTKFACE_TASK_NUM_CLASSES, LowLevelModel
    )
