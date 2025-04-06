import pytorch_lightning as pl
import torch

import wandb
from src.data.high_level_dm import HighLevelDataModule
from src.models.high_level_model import HighLevelModel


def train_high_level_model(filename):
    datamodule = HighLevelDataModule(root_dir="data", batch_size=64, num_workers=8)
    datamodule.setup()
    model = HighLevelModel()
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
    train_high_level_model("high-level-model")
