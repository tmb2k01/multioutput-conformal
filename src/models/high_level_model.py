import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class HighLevelModel(pl.LightningModule):
    """
    A multi-task classification model using a shared ResNet-50 feature extractor and separate heads per task.

    Each task is predicted independently using a dedicated classifier head.
    """

    def __init__(self, task_num_classes, classifier_heads=None, learning_rate=1e-3):
        """
        Args:
            task_num_classes (List[int]): Number of classes for each classification task.
            classifier_heads (Optional[ModuleList]): Custom classifier heads for each task.
            learning_rate (float): Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained ResNet50 and remove its classification head
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        in_features = base_model.fc.in_features
        self.task_num_classes = task_num_classes

        # Initialize separate classifier heads per task
        if classifier_heads is not None:
            self.classifiers = classifier_heads
        else:
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes),
                    )
                    for num_classes in task_num_classes
                ]
            )

        # Accuracy metric for each task
        self.acc_fn = nn.ModuleList(
            [
                torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=1
                )
                for num_classes in task_num_classes
            ]
        )

        self.loss_fn = F.cross_entropy
        self.test_step_outputs = []

    def forward(self, x):
        """
        Forward pass through the shared feature extractor and task-specific heads.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[Tensor]: List of output logits for each task.
        """
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return outputs

    def shared_step(self, batch, stage, accuracy=False):
        """
        Shared logic for training and validation steps.

        Args:
            batch (Tuple): Input batch (x, targets).
            stage (str): Either "train" or "val".
            accuracy (bool): Whether to compute and log accuracy.

        Returns:
            Tensor: Total loss across all tasks.
        """
        x, targets = batch
        outputs = self(x)

        losses = [self.loss_fn(out, target) for out, target in zip(outputs, targets)]
        total_loss = sum(losses)
        self.log(f"{stage}_loss", total_loss)

        if accuracy:
            for i, (out, target, acc_metric) in enumerate(
                zip(outputs, targets, self.acc_fn)
            ):
                acc = acc_metric(out, target)
                self.log(f"{stage}_acc_task{i}", acc)

            avg_acc = sum(
                [
                    acc_metric(out, target)
                    for acc_metric, out, target in zip(self.acc_fn, outputs, targets)
                ]
            ) / len(outputs)
            self.log(f"{stage}_acc", avg_acc)

        return total_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step with accuracy logging."""
        return self.shared_step(batch, "val", accuracy=True)

    def test_step(self, batch, batch_idx):
        """
        Test step: predicts task outputs and stores them for post-processing.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: True and predicted labels for each task.
        """
        x, targets = batch
        y_pred = self(x)

        true_labels = [target.cpu().numpy() for target in targets]
        pred_labels = [np.argmax(pred.cpu().numpy(), axis=1) for pred in y_pred]

        self.test_step_outputs.append((true_labels, pred_labels))
        return true_labels, pred_labels

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Computes and logs per-task metrics,
        as well as aggregate metrics over all tasks.
        """
        outputs = self.test_step_outputs

        true_labels = list(zip(*[out[0] for out in outputs]))
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
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
