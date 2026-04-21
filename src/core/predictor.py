from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from calibration.nonconformity_functions import NONCONFORMITY_FN_DIC
from core.calibrators import BaseCalibrator
from core.connector import (
    HighToHighLabelConnector,
    HighToLowLabelConnector,
    LowToHighLabelConnector,
    LowToLowLabelConnector,
)
from core.models import BaseModel
from core.types import NonconformityKey
from core.utils import expand_path
from data.datamodule import MultiOutputDataModule

import os
from pytorch_lightning.loggers import WandbLogger


def get_optional_wandb_logger(
    project: str,
    run_name: str | None = None,
    save_dir: str = ".",
):
    """
    Enable Weights & Biases only if a valid API key exists in .wandb_api_key.

    Looks for:
      ./.wandb_api_key

    Returns:
      WandbLogger or True logger fallback
    """
    key_file = Path(".wandb_api_key")

    if not key_file.exists():
        return True  # default Lightning logger

    api_key = key_file.read_text(encoding="utf-8").strip()

    # Basic validation: non-empty and not placeholder text
    if not api_key or api_key.lower() in {"none", "null", "your_key_here"}:
        return True

    os.environ["WANDB_API_KEY"] = api_key

    try:
        import wandb

        wandb.login(key=api_key, relogin=True)
        return WandbLogger(
            project=project,
            name=run_name,
            save_dir=save_dir,
            log_model=False,
        )
    except Exception as e:
        print(f"W&B disabled: {e}")
        return True

@dataclass
class ConformalPredictor:
    """Train, save, load, and run conformal prediction.

    Stores CP configuration at init:
      - nonconformity_key: which nonconformity score to compute
      - cp_type: which CP method (threshold key) to use at inference

    Also stores the last computed nonconformity scores in `last_nonconformity_scores`.
    """

    # core
    task_num_classes: list[int]
    model: BaseModel
    calibrator: BaseCalibrator

    # conformal configuration (stored at init)
    nonconformity_key: NonconformityKey = "hinge"
    cp_type: str = "scp_global_threshold"  # calibration_key stored in thresholds json

    # runtime / IO
    artifacts_dir: Path = field(default_factory=lambda: Path("./artifacts"))

    def __post_init__(self) -> None:
        self.artifacts_dir = expand_path(self.artifacts_dir)
        if self.nonconformity_key not in NONCONFORMITY_FN_DIC:
            raise ValueError(
                f"Unknown nonconformity_key {self.nonconformity_key!r}. "
                f"Available: {sorted(NONCONFORMITY_FN_DIC.keys())}"
            )

        if self.model.level != self.calibrator.level:
            if self.model.level == "high":
                self.connector = HighToLowLabelConnector(self.task_num_classes)
            else:
                self.connector = LowToHighLabelConnector(self.task_num_classes)
        else:
            if self.model.level == "high":
                self.connector = HighToHighLabelConnector(self.task_num_classes)
            else:
                self.connector = LowToLowLabelConnector(self.task_num_classes)

    # -----------------------------
    # constructors
    # -----------------------------
    @staticmethod
    def build(
        model_cls: type[BaseModel],
        calibrator_cls: type[BaseCalibrator],
        task_num_classes: list[int],
        learning_rate: float = 1e-3,
        artifacts_dir: str | Path = "./artifacts",
        nonconformity_key: NonconformityKey = "hinge",
        cp_type: str = "scp_global_threshold",
    ) -> ConformalPredictor:
        model = model_cls(
            task_num_classes=task_num_classes, learning_rate=learning_rate
        )
        calibrator = calibrator_cls(calibrationFnKey=cp_type,
                                    nonconformityFnKey=nonconformity_key,
                                    load_on_init=False,
                                    artifacts_dir=artifacts_dir)

        return ConformalPredictor(
            task_num_classes=list(task_num_classes),
            model=model,
            calibrator=calibrator,
            artifacts_dir=expand_path(artifacts_dir),
            nonconformity_key=nonconformity_key,
            cp_type=cp_type,
        )

    @staticmethod
    def load(
        model_cls: type[BaseModel],
        calibrator_cls: type[BaseCalibrator],
        task_num_classes: list[int],
        alpha: float = 0.05,
        artifacts_dir: str | Path = "./artifacts",
        nonconformity_key: NonconformityKey = "hinge",
        cp_type: str = "scp_global_threshold",
    ) -> ConformalPredictor:
        """Load a previously saved model checkpoint and calibration thresholds."""
        artifacts_dir = expand_path(artifacts_dir)
        ckpt = artifacts_dir / "models" / f"{model_cls.level}-model.ckpt"
        thr = artifacts_dir / "thresholds" / f"{calibrator_cls.level}" / f"{nonconformity_key}" / f"{cp_type}" / f"alpha_{alpha:.2f}.json"

        if not ckpt.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt}")
        load_on_init = thr.is_file()
        model = model_cls.load_from_checkpoint(
            str(ckpt), task_num_classes=task_num_classes
        )
        model.eval()
        calibrator = calibrator_cls(calibrationFnKey=cp_type,
                                    nonconformityFnKey=nonconformity_key,
                                    load_on_init=load_on_init,
                                    alpha=alpha,
                                    artifacts_dir=artifacts_dir,
                                    )

        return ConformalPredictor(
            task_num_classes=list(task_num_classes),
            model=model,
            calibrator=calibrator,
            artifacts_dir=artifacts_dir,
            nonconformity_key=nonconformity_key,
            cp_type=cp_type,
        )

    # -----------------------------
    # sklearn-like interface
    # -----------------------------
    def fit(
        self,
        data_module: MultiOutputDataModule,
        alpha: float = 0.05,
        calibration_clusters: str | int = "auto",
        max_epochs: int = 30,
        train_model: bool = True,
        calibrate_model: bool = True,
    ) -> ConformalPredictor:
        """Train (optional) and calibrate; mirrors train.py."""
        if train_model and calibrate_model:
            self._train_and_calibrate(
                data_module=data_module,
                alpha=alpha,
                calibration_clusters=calibration_clusters,
                max_epochs=max_epochs,
            )
        elif train_model:
            self._train(
                data_module=data_module,
                max_epochs=max_epochs,
            )
        elif calibrate_model:
            self._calibrate(
                data_module=data_module,
                alpha=alpha,
                calibration_clusters=calibration_clusters,
            )
        return self


    # -----------------------------
    # internal helpers
    # -----------------------------

    def _prepare_artifact_dir(self) -> None:
        self.artifacts_dir = expand_path(self.artifacts_dir)


    def _make_training_trainer(self, max_epochs: int) -> tuple[pl.Trainer, pl.callbacks.ModelCheckpoint]:
        models_dir = self.artifacts_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        ckpt_cb = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(models_dir),
            filename=f"{self.model.level}-model",
            save_top_k=1,
            save_weights_only=False,
            mode="min",
        )
        es_cb = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=True,
            mode="min",
        )

        logger = get_optional_wandb_logger(
            project="multioutput-conformal-prediction",
            run_name="model-training",
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[es_cb, ckpt_cb],
            logger=logger,
        )
        return trainer, ckpt_cb


    def _make_prediction_trainer(self) -> pl.Trainer:
        return pl.Trainer(logger=False)


    def _reload_best_model(self, ckpt_path: str) -> None:
        self.model_ckpt_path = Path(ckpt_path)
        model_cls = self.model.__class__
        self.model = model_cls.load_from_checkpoint(
            ckpt_path,
            task_num_classes=self.task_num_classes,
        )
        self.model.eval()


    def _train_model(self, data_module: MultiOutputDataModule, max_epochs: int) -> None:
        trainer, ckpt_cb = self._make_training_trainer(max_epochs=max_epochs)
        trainer.fit(self.model, data_module)

        if not ckpt_cb.best_model_path:
            raise RuntimeError("No best_model_path produced by ModelCheckpoint.")

        self._reload_best_model(ckpt_cb.best_model_path)


    def _get_calibration_preds_and_labels(
        self,
        data_module: MultiOutputDataModule,
        trainer: pl.Trainer | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if trainer is None:
            trainer = self._make_prediction_trainer()

        self.model.eval()

        preds = trainer.predict(self.model, dataloaders=data_module.calib_dataloader())
        preds = self.connector.pred_to_calib(preds)

        labels = self.get_labels(data_module.datasets["calib"])

        return preds, labels


    def _fit_calibrator(
        self,
        data_module: MultiOutputDataModule,
        alpha: float,
        calibration_clusters: str | int,
        trainer: pl.Trainer | None = None,
    ) -> None:
        thresholds_dir = self.artifacts_dir / "thresholds"
        thresholds_dir.mkdir(parents=True, exist_ok=True)

        preds, labels = self._get_calibration_preds_and_labels(
            data_module=data_module,
            trainer=trainer,
        )

        self.calibrator.fit(
            preds,
            labels,
            alpha=alpha,
            n_clusters=calibration_clusters,
        )


    # -----------------------------
    # workflow methods
    # -----------------------------

    def _calibrate(
        self,
        data_module: MultiOutputDataModule,
        alpha: float,
        calibration_clusters: str | int,
    ) -> None:
        """Calibrate only (no model training)."""
        self._prepare_artifact_dir()
        self._fit_calibrator(
            data_module=data_module,
            alpha=alpha,
            calibration_clusters=calibration_clusters,
        )


    def _train(
        self,
        data_module: MultiOutputDataModule,
        max_epochs: int,
    ) -> None:
        """Train only."""
        self._prepare_artifact_dir()
        self._train_model(
            data_module=data_module,
            max_epochs=max_epochs,
        )


    def _train_and_calibrate(
        self,
        data_module: MultiOutputDataModule,
        alpha: float,
        calibration_clusters: str | int,
        max_epochs: int,
    ) -> None:
        """Train and then calibrate."""
        self._prepare_artifact_dir()
        self._train_model(
            data_module=data_module,
            max_epochs=max_epochs,
        )
        self._fit_calibrator(
            data_module=data_module,
            alpha=alpha,
            calibration_clusters=calibration_clusters,
        )

    # -----------------------------
    # inference
    # -----------------------------
    def predict(self, data_loader: torch.utils.data.DataLoader) -> Any:
        """Compute prediction sets using init-time CP configuration."""
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(device)
        self.model.eval()

        trainer = self._make_prediction_trainer()
        preds = trainer.predict(self.model, dataloaders=data_loader)

        outputs = self.connector.pred_to_calib(preds)
        return self.calibrator.predict(outputs)
    
    def get_labels(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        labels = np.stack(
            [labels for _, labels in dataset], axis=1
        )
        return self.connector.gt_to_calib(labels)
