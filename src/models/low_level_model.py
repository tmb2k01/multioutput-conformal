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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, _ = batch  # We don't use targets during prediction
        outputs = self(x)
        return torch.softmax(outputs, dim=1)

    def encode_targets(self, targets):
        """
        Transforms multi-task targets into a single joint class ID.

        Args:
            targets (Tensor or List[Tensor]): Either a (B, T) tensor or list of Tensors of shape (B,).

        Returns:
            Tensor: Encoded target vector (B,).
        """
        # If input is a single Tensor of shape (B, T), use it directly
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2:
                pass  # already correct shape
            else:
                raise ValueError(
                    f"Expected targets of shape (B, T), got shape {targets.shape}"
                )
        # If input is a list or tuple of Tensors, stack along the last dim
        elif isinstance(targets, (list, tuple)):
            targets = torch.stack(targets, dim=1)
        else:
            raise TypeError(f"Unsupported target type: {type(targets)}")

        targets = targets.to(self.multipliers.device)
        target = (targets * self.multipliers).sum(dim=1).long()
        return target

    def decode_targets(self, joint_class):
        """
        Decodes joint class IDs into separate task-specific labels using modular arithmetic.

        Args:
            joint_class (ndarray): A scalar or NumPy array containing encoded class indices.

        Returns:
            Tensor: A 2D tensor of shape (N, num_tasks), where N is the number of samples.
        """
        if joint_class.ndim == 0:
            joint_class = np.array([joint_class])

        labels = []
        for num_classes in reversed(self.task_num_classes):
            labels.append(joint_class % num_classes)
            joint_class = joint_class // num_classes

        stacked = np.stack(labels[::-1], axis=-1)
        return [stacked[:, i] for i in range(stacked.shape[1])]

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
        target = self.encode_targets(targets)
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
        targets = targets.T  # shape [T, B]
        y_pred = self(x)

        true_labels = [target.cpu().numpy() for target in targets]
        pred_labels = np.argmax(y_pred.cpu().numpy(), axis=1)

        pred_labels = self.decode_targets(pred_labels)
        self.test_step_outputs.append((true_labels, pred_labels))
        return true_labels, pred_labels

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Computes and logs per-task metrics,
        as well as aggregate metrics over all tasks.
        """
        outputs = self.test_step_outputs

        # Group by task across batches
        true_labels = list(
            zip(*[out[0] for out in outputs])
        )  # List of length T, each of shape (B,)
        pred_labels = list(zip(*[out[1] for out in outputs]))

        task_accuracies = []
        all_true = []
        all_pred = []

        for i, (y_true, y_pred) in enumerate(zip(true_labels, pred_labels)):
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            all_true.append(y_true)
            all_pred.append(y_pred)

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_true, y_pred)
            conf_acc = np.trace(conf_matrix) / np.sum(conf_matrix)

            task_accuracies.append(accuracy)

            self.log(f"test_task_{i}_accuracy", accuracy)
            self.log(f"test_task_{i}_f1", f1)
            self.log(f"test_task_{i}_conf_acc", conf_acc)

            print(f"Task {i} Accuracy: {accuracy:.2f}")
            print(f"Task {i} F1 Score: {f1:.2f}")
            print(f"Task {i} Confusion Matrix:")
            print(conf_matrix)

        all_true_stacked = np.stack(all_true, axis=1)
        all_pred_stacked = np.stack(all_pred, axis=1)

        correct_per_sample = np.all(all_true_stacked == all_pred_stacked, axis=1)
        overall_accuracy = np.mean(correct_per_sample)

        self.log("test_accuracy (Low-level)", overall_accuracy)
        self.log("test_accuracy (High-level)", np.mean(task_accuracies))

        print(f"Overall Accuracy (Low-level): {overall_accuracy:.2f}")
        print(f"Mean Task Accuracy (High-level): {np.mean(task_accuracies):.2f}")

        self.test_step_outputs = []

    def configure_optimizers(self):
        """
        Sets up the optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
