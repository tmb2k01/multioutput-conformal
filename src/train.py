import pytorch_lightning as pl
import torch

import wandb
from src.data.high_level_dm import HighLevelDataModule
from src.models.high_level_model import HighLevelModel

# MDC Dataset properties
MDC_COLOR = 12
MDC_TYPE = 11
MDC_TASK_NUM_CLASSES = [MDC_COLOR, MDC_TYPE]


def train_high_level_model(root_dir, filename, task_num_classes):
    datamodule = HighLevelDataModule(
        root_dir=root_dir,
        task_num_classes=task_num_classes,
        batch_size=64,
        num_workers=8,
    )
    datamodule.setup()
    model = HighLevelModel(task_num_classes=task_num_classes)
    wandb_logger = pl.loggers.WandbLogger(
        project=f"{filename}",
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
        filename=filename,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
    )
    trainer = pl.Trainer(
        max_epochs=30, callbacks=[early_stopping, checkpoint], logger=wandb_logger
    )
    trainer.fit(model, datamodule)


def train():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    wandb.login()
    train_high_level_model("data", "mdc-high-level-model", MDC_TASK_NUM_CLASSES)
