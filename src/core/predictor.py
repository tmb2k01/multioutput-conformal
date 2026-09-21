from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

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
from core.utils import convert_multitask_preds, expand_path, to_numpy
from data.datamodule import MultiOutputDataModule


def get_optional_wandb_logger(
    project: str,
    run_name: str | None = None,
    save_dir: str = ".",
) -> WandbLogger | True:
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
    model_ckpt_path: Path | None = field(default=None, init=False, repr=False)

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
        calibrator = calibrator_cls(calibration_fn_key=cp_type,
                                    nonconformity_fn_key=nonconformity_key,
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
        thr = (
            artifacts_dir
            / "thresholds"
            / f"{calibrator_cls.level}"
            / f"{nonconformity_key}"
            / f"{cp_type}"
            / f"alpha_{alpha:.2f}.json"
        )

        if not ckpt.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt}")
        load_on_init = thr.is_file()
        model = model_cls.load_from_checkpoint(
            str(ckpt), task_num_classes=task_num_classes
        )
        model.eval()
        calibrator = calibrator_cls(calibration_fn_key=cp_type,
                                    nonconformity_fn_key=nonconformity_key,
                                    load_on_init=load_on_init,
                                    alpha=alpha,
                                    artifacts_dir=artifacts_dir,
                                    )

        predictor = ConformalPredictor(
            task_num_classes=list(task_num_classes),
            model=model,
            calibrator=calibrator,
            artifacts_dir=artifacts_dir,
            nonconformity_key=nonconformity_key,
            cp_type=cp_type,
        )
        predictor.model_ckpt_path = ckpt
        return predictor

    # -----------------------------
    # sklearn-like interface
    # -----------------------------
    def train(
        self,
        data_module: MultiOutputDataModule,
        max_epochs: int = 30,
    ) -> ConformalPredictor:
        """Train the model only and save its checkpoint under ``artifacts_dir``."""
        self._prepare_artifact_dir()
        self._train_model(data_module=data_module, max_epochs=max_epochs)
        return self

    def calibrate(
        self,
        data_module: MultiOutputDataModule,
        alpha: float = 0.05,
        calibration_clusters: str | int = "auto",
    ) -> ConformalPredictor:
        """Calibrate the current model only and save thresholds under ``artifacts_dir``.

        Expects a trained model (e.g. loaded via :meth:`load` from ``artifacts_dir``).
        """
        self._prepare_artifact_dir()
        self._fit_calibrator(
            data_module=data_module,
            alpha=alpha,
            calibration_clusters=calibration_clusters,
        )
        return self

    def fit(
        self,
        data_module: MultiOutputDataModule,
        alpha: float = 0.05,
        calibration_clusters: str | int = "auto",
        max_epochs: int = 30,
        train_model: bool = True,
        calibrate_model: bool = True,
    ) -> ConformalPredictor:
        """Convenience wrapper to train and/or calibrate in one call."""
        if train_model:
            self.train(data_module=data_module, max_epochs=max_epochs)
        if calibrate_model:
            self.calibrate(
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


    def _make_training_trainer(
        self, max_epochs: int
    ) -> tuple[pl.Trainer, pl.callbacks.ModelCheckpoint]:
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

    def _model_output_cache_path(
        self,
        data_loader: torch.utils.data.DataLoader,
        cache_name: str,
    ) -> Path | None:
        if self.model_ckpt_path is None or not self.model_ckpt_path.is_file():
            return None

        digest = hashlib.sha256()
        checkpoint_stat = self.model_ckpt_path.stat()
        digest.update(str(self.model_ckpt_path.resolve()).encode())
        digest.update(f":{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}".encode())
        digest.update(self.model.__class__.__qualname__.encode())
        digest.update(repr(self.task_num_classes).encode())

        dataset = data_loader.dataset
        digest.update(dataset.__class__.__qualname__.encode())
        digest.update(str(len(dataset)).encode())
        digest.update(repr(getattr(dataset, "transform", None)).encode())

        samples = getattr(dataset, "samples", None)
        if samples is None:
            return None

        # Dataset splits are immutable during an experiment. Hashing paths keeps
        # split identity without issuing one filesystem stat call per image.
        for image_path, _ in samples:
            digest.update(os.path.abspath(image_path).encode())

        cache_dir = self.artifacts_dir / "model_outputs"
        return cache_dir / f"{cache_name}-{digest.hexdigest()[:20]}.npz"

    def _load_model_output_cache(
        self,
        cache_path: Path,
        expected_samples: int,
    ) -> list[np.ndarray] | None:
        if not cache_path.is_file():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                output_keys = sorted(
                    (key for key in cached.files if key.startswith("output_")),
                    key=lambda key: int(key.removeprefix("output_")),
                )
                outputs = [cached[key].copy() for key in output_keys]
        except (OSError, ValueError, KeyError):
            return None

        expected_outputs = len(self.task_num_classes) if self.model.level == "high" else 1
        if (
            len(outputs) != expected_outputs
            or any(output.shape[0] != expected_samples for output in outputs)
        ):
            return None

        print(f"Using cached model outputs: {cache_path}")
        return outputs

    def _save_model_output_cache(
        self,
        cache_path: Path,
        outputs: list[np.ndarray],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            **{
                f"output_{index}": output
                for index, output in enumerate(outputs)
            },
        )

        print(f"Saved model outputs: {cache_path}")

    def _run_model_prediction(
        self,
        data_loader: torch.utils.data.DataLoader,
        trainer: pl.Trainer,
    ) -> list[np.ndarray]:
        predictions = trainer.predict(self.model, dataloaders=data_loader)
        if self.model.level == "high":
            return convert_multitask_preds(predictions)

        return [
            np.concatenate([to_numpy(batch) for batch in predictions], axis=0)
        ]

    def _get_model_outputs(
        self,
        data_loader: torch.utils.data.DataLoader,
        trainer: pl.Trainer,
        cache_name: str | None = None,
    ) -> list[list[np.ndarray]] | list[np.ndarray]:
        cache_path = (
            self._model_output_cache_path(data_loader, cache_name)
            if cache_name is not None
            else None
        )

        if cache_path is None:
            outputs = self._run_model_prediction(data_loader, trainer)
        else:
            outputs = self._load_model_output_cache(cache_path, len(data_loader.dataset))
            if outputs is None:
                print(f"Computing model outputs: {cache_path}")
                outputs = self._run_model_prediction(data_loader, trainer)
                self._save_model_output_cache(cache_path, outputs)

        if self.model.level == "high":
            return [outputs]
        return outputs


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

        preds = self._get_model_outputs(
            data_module.calib_dataloader(),
            trainer,
            cache_name=(
                f"calib_{data_module.iter}"
                if data_module.iter is not None
                else "calib"
            ),
        )
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
    # inference
    # -----------------------------
    def predict(
        self,
        data_loader: torch.utils.data.DataLoader,
        cache_name: str | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Compute prediction sets using init-time CP configuration."""
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(device)
        self.model.eval()

        trainer = self._make_prediction_trainer()
        preds = self._get_model_outputs(data_loader, trainer, cache_name=cache_name)

        outputs = self.connector.pred_to_calib(preds)
        return self.calibrator.predict(outputs)
    
    def get_labels(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        labels = np.stack(
            [labels for _, labels in dataset], axis=1
        )
        return self.connector.gt_to_calib(labels)
