import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class LowLevelModel(pl.LightningModule):
    """
    A PyTorch Lightning module that maps multi-task classification targets
    to a single joint class using a ResNet50 backbone for feature extraction.
    """

    def __init__(self, task_num_classes, classifier_head=None, learning_rate=1e-3):
        """
        Args:
            task_num_classes (List[int]): A list specifying the number of classes for each task.
            classifier_head (nn.Module, optional): Custom classifier head. If None, a default MLP head is used.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # Use pretrained ResNet50 as feature extractor (frozen)
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        in_features = base_model.fc.in_features
        self.task_num_classes = task_num_classes

        # Create multipliers to encode multi-task targets into a single class ID
        self.register_buffer(
            "multipliers",
            torch.tensor(
                [
                    math.prod(self.task_num_classes[i + 1 :])
                    for i in range(len(self.task_num_classes))
                ],
                dtype=torch.long,
            ),
        )

        num_classes = math.prod(task_num_classes)

        # Classifier head
        if classifier_head is not None:
            self.classifier = classifier_head
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        # Metrics and loss
        self.acc_fn = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.loss_fn = F.cross_entropy
        self.test_step_outputs = []

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input images.

        Returns:
            Tensor: Logits for each joint class.
        """
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output

    def transform_targets(self, targets):
        """
        Transforms multi-task targets into a single joint class ID.

        Args:
            targets (List[Tensor]): List of tensors, each representing a task label.

        Returns:
            Tensor: Encoded target vector (batch_size,).
        """
        targets = torch.stack(targets, dim=1)  # [batch_size, num_tasks]
        targets = targets.to(self.multipliers.device)
        target = (targets * self.multipliers).sum(dim=1).long()
        return target

    def shared_step(self, batch, stage, accuracy=False):
        """
        Shared logic for training/validation.

        Args:
            batch (Tuple): Input data and targets.
            stage (str): Current phase ('train' or 'val').
            accuracy (bool): Whether to compute accuracy.

        Returns:
            Tensor: Loss value.
        """
        x, targets = batch
        target = self.transform_targets(targets)
        output = self(x)
        loss = self.loss_fn(output, target)
        self.log(f"{stage}_loss", loss)

        if accuracy:
            acc = self.acc_fn(output, target)
            self.log(f"{stage}_acc", acc)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", accuracy=True)

    def test_step(self, batch, batch_idx):
        """
        Stores true and predicted labels for evaluation after test epoch.
        """
        x, targets = batch
        target = self.transform_targets(targets)
        y_pred = self(x)
        pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
        self.test_step_outputs.append((target.cpu().numpy(), pred_labels))
        return target, pred_labels

    def on_test_epoch_end(self):
        """
        Calculates and logs metrics at the end of the test epoch.
        """
        outputs = self.test_step_outputs
        true_labels = np.concatenate([out[0] for out in outputs])
        pred_labels = np.concatenate([out[1] for out in outputs])

        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")

        self.log("test_accuracy", accuracy)
        self.log("test_f1", f1)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")

        self.test_step_outputs = []

    def configure_optimizers(self):
        """
        Sets up the optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
