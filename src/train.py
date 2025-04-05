import pytorch_lightning as pl

from src.data.high_level_dm import HighLevelDataModule
from src.models.high_level_model import HighLevelModel


def train():
    datamodule = HighLevelDataModule(root_dir="data", batch_size=32, num_workers=4)
    datamodule.setup()
    model = HighLevelModel()
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_acc",
        patience=3,
        verbose=True,
        mode="min",
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath="models/",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(max_epochs=20, callbacks=[early_stopping, checkpoint])
    trainer.fit(model, datamodule)
