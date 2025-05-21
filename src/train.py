import json
import math
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
    load_preds: bool = False,
    alpha: float = 0.05,
    clusters=None,
):
    high_level = isinstance(model, HighLevelModel)
    level = "high" if high_level else "low"
    preds_path = f"./models/sgvehicle-{level}-model-calibpreds.npz"

    if load_preds:
        loaded = np.load(preds_path)
        calib_preds = [loaded[key] for key in loaded.files]
        if not high_level:
            calib_preds = np.array(calib_preds)
    else:
        model.eval()
        trainer = pl.Trainer(accelerator="gpu")
        preds = trainer.predict(model, dataloaders=datamodule.calib_dataloader())

        if high_level:
            calib_preds = convert_multitask_preds(preds)
        else:
            calib_preds = np.concatenate(preds, axis=0)
            calib_preds = np.array(calib_preds)

        np.savez(preds_path, *calib_preds)

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
        clusters=clusters,
    )

    with open(f"./models/sgvehicle-{level}-level-calibration.json", "w") as f:
        json.dump(convert_numpy_to_native(q_hats), f, indent=2)


def train_model(
    root_dir,
    filename,
    task_num_classes,
    model: Union[HighLevelModel, LowLevelModel],
    alpha: float = 0.05,
    calibration_clusters: Union[None, int] = None,
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
        monitor="val_acc",
        patience=5,
        verbose=True,
        mode="max",
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath="models/",
        filename=f"{filename}-model",
        save_top_k=1,
        save_weights_only=False,
        mode="max",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=30,
        callbacks=[early_stopping, checkpoint],
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule)

    # Calibration logic goes here
    model.eval()
    trainer = pl.Trainer(accelerator="gpu")

    calibrate_model(
        model,
        datamodule,
        load_preds=True,
        alpha=alpha,
        clusters=calibration_clusters,
    )


def train():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    wandb.login()
    train_model("data", "sgvehicle-high-level", SGVEHICLE_TASK_NUM_CLASSES, HighLevelModel)

    train_model("data", "sgvehicle-low-level", SGVEHICLE_TASK_NUM_CLASSES, LowLevelModel)
