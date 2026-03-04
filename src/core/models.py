from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# -----------------------------
# Base (OOP) building blocks
# -----------------------------
class BaseModel(pl.LightningModule, ABC):
    """
    OOP refactor:
      - BaseModel owns shared lifecycle (optimizer, common epoch-end evaluation, prediction step defaults)
      - Subclasses implement only what differs:
          * level
          * forward_logits (model logits)
          * training/validation loss + metric computation hooks
          * test decoding strategy (how to turn logits into per-task predictions)
    """

    # ---- identity / metadata ----
    level: str

    def __init__(self, task_num_classes: Sequence[int], learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor"])
        self.task_num_classes: list[int] = list(task_num_classes)
        self.learning_rate = float(learning_rate)

        # Shared loss
        self.loss_fn = F.cross_entropy

        # Collected at test time: list of (true_per_task: List[np.ndarray], pred_per_task: List[np.ndarray])
        self._test_step_outputs: list[tuple[list[np.ndarray], list[np.ndarray]]] = []

    

    # ---- core forward API ----
    @abstractmethod
    def forward_logits(self, x: torch.Tensor) -> Any:
        """Return logits (either Tensor for low-level or List[Tensor] for high-level)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Any:
        return self.forward_logits(x)

    # ---- training/validation hooks ----
    @abstractmethod
    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, accuracy: bool) -> torch.Tensor:
        """Compute loss (and log metrics if requested) for train/val."""
        raise NotImplementedError

    # ---- test decoding hooks ----
    @abstractmethod
    def _predict_proba_for_batch(self, x: torch.Tensor) -> Any:
        """Return probabilities (softmax outputs), shape mirrors forward output type."""
        raise NotImplementedError

    @abstractmethod
    def _decode_predictions_for_test(self, logits_or_proba: Any) -> list[np.ndarray]:
        """
        Convert model outputs into per-task predicted labels:
          - returns List[np.ndarray] length T, each shape (B,)
        """
        raise NotImplementedError

    # ---- lightning steps ----
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train", accuracy=False)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val", accuracy=True)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        x, _ = batch
        return self._predict_proba_for_batch(x)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        x, targets = batch

        # targets expected as (B, T)
        if not isinstance(targets, torch.Tensor) or targets.ndim != 2:
            raise ValueError(f"Expected targets of shape (B, T) tensor, got {type(targets)} with shape {getattr(targets, 'shape', None)}")

        # True labels per task (List[np.ndarray] of shape (B,))
        targets_t = targets.T  # (T,B)
        true_labels: list[np.ndarray] = [t.detach().cpu().numpy() for t in targets_t]

        # Pred labels per task
        with torch.no_grad():
            outputs = self.forward_logits(x)
        pred_labels: list[np.ndarray] = self._decode_predictions_for_test(outputs)

        self._test_step_outputs.append((true_labels, pred_labels))
        return true_labels, pred_labels

    def on_test_epoch_end(self) -> None:
        """
        Unified test aggregation for both models:
          - per-task accuracy, f1, confusion-matrix accuracy
          - "Low-level" sample-wise all-tasks-correct accuracy
          - "High-level" mean task accuracy
        """
        outputs = self._test_step_outputs
        if not outputs:
            return

        # group by task across batches
        true_by_task = list(zip(*[out[0] for out in outputs], strict=False))  # T lists of arrays
        pred_by_task = list(zip(*[out[1] for out in outputs], strict=False))

        task_accuracies: list[float] = []
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []

        for i, (y_true_batches, y_pred_batches) in enumerate(zip(true_by_task, pred_by_task, strict=False)):
            y_true = np.concatenate(list(y_true_batches))
            y_pred = np.concatenate(list(y_pred_batches))

            all_true.append(y_true)
            all_pred.append(y_pred)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            cm = confusion_matrix(y_true, y_pred)
            cm_acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0

            task_accuracies.append(float(acc))

            self.log(f"test_task_{i}_accuracy", float(acc))
            self.log(f"test_task_{i}_f1", float(f1))
            self.log(f"test_task_{i}_conf_acc", cm_acc)

            print(f"Task {i} Accuracy: {acc:.2f}")
            print(f"Task {i} F1 Score: {f1:.2f}")
            print(f"Task {i} Confusion Matrix:")
            print(cm)

        all_true_stacked = np.stack(all_true, axis=1)  # (N,T)
        all_pred_stacked = np.stack(all_pred, axis=1)  # (N,T)

        correct_per_sample = np.all(all_true_stacked == all_pred_stacked, axis=1)
        overall_accuracy = float(np.mean(correct_per_sample))
        mean_task_accuracy = float(np.mean(task_accuracies))

        self.log("test_accuracy (Low-level)", overall_accuracy)
        self.log("test_accuracy (High-level)", mean_task_accuracy)

        print(f"Overall Accuracy (Low-level): {overall_accuracy:.2f}")
        print(f"Mean Task Accuracy (High-level): {mean_task_accuracy:.2f}")

        self._test_step_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    # ---- shared utilities for subclasses ----
    @staticmethod
    def build_frozen_resnet50_feature_extractor() -> tuple[nn.Module, int]:
        """
        Returns:
          feature_extractor: ResNet50 without final fc (outputs (B, 2048, 1, 1))
          in_features: 2048 (fc input size)
        """
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(base.children())[:-1])
        for p in feature_extractor.parameters():
            p.requires_grad = False
        return feature_extractor, base.fc.in_features


# -----------------------------
# High-level model (multi-head)
# -----------------------------
class HighLevelModel(BaseModel):
    """
    Multi-task classification:
      - shared frozen ResNet50 features
      - separate classifier heads per task
    """

    level = "high"

    def __init__(
        self,
        task_num_classes: Sequence[int],
        classifier_heads: nn.ModuleList | None = None,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(task_num_classes=task_num_classes, learning_rate=learning_rate)

        self.feature_extractor, in_features = self.build_frozen_resnet50_feature_extractor()

        if classifier_heads is not None:
            self.classifiers = classifier_heads
        else:
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, n),
                    )
                    for n in self.task_num_classes
                ]
            )

        self.acc_fn = nn.ModuleList(
            [torchmetrics.Accuracy(task="multiclass", num_classes=n, top_k=1) for n in self.task_num_classes]
        )

    def forward_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.feature_extractor(x)
        feats = torch.flatten(feats, 1)
        return [head(feats) for head in self.classifiers]

    def _predict_proba_for_batch(self, x: torch.Tensor) -> list[torch.Tensor]:
        logits = self.forward_logits(x)
        return [torch.softmax(l, dim=1) for l in logits]

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, accuracy: bool) -> torch.Tensor:
        x, targets = batch  # targets: (B,T)
        logits = self.forward_logits(x)  # list of (B, C_t)

        if not isinstance(targets, torch.Tensor) or targets.ndim != 2:
            raise ValueError(f"Expected targets of shape (B, T), got {type(targets)} with shape {getattr(targets,'shape',None)}")

        targets_t = targets.T  # (T,B)

        losses = [self.loss_fn(logit, targets_t[i]) for i, logit in enumerate(logits)]
        total_loss = sum(losses)
        self.log(f"{stage}_loss", total_loss)

        if accuracy:
            per_task_acc = []
            for i, (logit, t, metric) in enumerate(zip(logits, targets_t, self.acc_fn, strict=False)):
                acc = metric(logit, t)
                self.log(f"{stage}_acc_task{i}", acc)
                per_task_acc.append(acc)
            if per_task_acc:
                self.log(f"{stage}_acc", sum(per_task_acc) / len(per_task_acc))

        return total_loss

    def _decode_predictions_for_test(self, logits_or_proba: Any) -> list[np.ndarray]:
        # expects list[Tensor] logits
        if not isinstance(logits_or_proba, (list, tuple)):
            raise TypeError("HighLevelModel expects list/tuple outputs.")
        preds: list[np.ndarray] = []
        for out in logits_or_proba:
            pred = torch.argmax(out, dim=1).detach().cpu().numpy()
            preds.append(pred)
        return preds


# -----------------------------
# Low-level model (joint class)
# -----------------------------
class LowLevelModel(BaseModel):
    """
    Single-head joint classification:
      - shared frozen ResNet50 features
      - one classifier head over product(task_num_classes)
      - encode/decode utilities
    """

    level = "low"

    def __init__(
        self,
        task_num_classes: Sequence[int],
        classifier_head: nn.Module | None = None,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(task_num_classes=task_num_classes, learning_rate=learning_rate)

        self.feature_extractor, in_features = self.build_frozen_resnet50_feature_extractor()

        # multipliers for joint encoding
        multipliers = torch.tensor(
            [math.prod(self.task_num_classes[i + 1 :]) for i in range(len(self.task_num_classes))],
            dtype=torch.long,
        )
        self.register_buffer("multipliers", multipliers)

        self.num_classes = int(math.prod(self.task_num_classes))

        self.classifier = classifier_head or nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

        self.acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        feats = torch.flatten(feats, 1)
        return self.classifier(feats)

    def _predict_proba_for_batch(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return torch.softmax(logits, dim=1)

    def encode_targets(self, targets: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """
        targets:
          - Tensor (B,T) OR
          - sequence of T tensors each (B,)
        returns: Tensor (B,) joint ids
        """
        if isinstance(targets, torch.Tensor):
            if targets.ndim != 2:
                raise ValueError(f"Expected targets of shape (B, T), got {tuple(targets.shape)}")
            t = targets
        elif isinstance(targets, (list, tuple)):
            t = torch.stack(list(targets), dim=1)
        else:
            raise TypeError(f"Unsupported targets type: {type(targets)}")

        t = t.to(self.multipliers.device).long()
        return (t * self.multipliers[None, :]).sum(dim=1).long()

    def decode_joint(self, joint: np.ndarray) -> list[np.ndarray]:
        """
        joint: (N,) array of joint ids
        returns: List of length T, each (N,) task label
        """
        if joint.ndim == 0:
            joint = np.array([joint])

        labels = []
        x = joint.copy()
        for n in reversed(self.task_num_classes):
            labels.append(x % n)
            x = x // n

        stacked = np.stack(labels[::-1], axis=-1)  # (N,T)
        return [stacked[:, i] for i in range(stacked.shape[1])]

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, accuracy: bool) -> torch.Tensor:
        x, targets = batch  # targets: (B,T)
        joint_target = self.encode_targets(targets)
        logits = self.forward_logits(x)

        loss = self.loss_fn(logits, joint_target)
        self.log(f"{stage}_loss", loss)

        if accuracy:
            acc = self.acc_fn(logits, joint_target)
            self.log(f"{stage}_acc", acc)

        return loss

    def _decode_predictions_for_test(self, logits_or_proba: Any) -> list[np.ndarray]:
        # expects Tensor logits (B, num_classes)
        if not isinstance(logits_or_proba, torch.Tensor):
            raise TypeError("LowLevelModel expects Tensor outputs.")
        joint_pred = torch.argmax(logits_or_proba, dim=1).detach().cpu().numpy()  # (B,)
        return self.decode_joint(joint_pred)