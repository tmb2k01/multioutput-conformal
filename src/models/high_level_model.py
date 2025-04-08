import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class HighLevelModel(pl.LightningModule):
    def __init__(self, task_num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        in_features = base_model.fc.in_features

        # Dynamically create classifier heads
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

        # One accuracy metric per task
        self.accuracies = nn.ModuleList(
            [
                torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                for num_classes in task_num_classes
            ]
        )

        self.loss_fn = F.cross_entropy

        self.test_step_outputs = []

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return outputs

    def shared_step(self, batch, stage, accuracy=False):
        x, targets = batch
        outputs = self(x)

        losses = [self.loss_fn(out, target) for out, target in zip(outputs, targets)]
        total_loss = sum(losses)
        self.log(f"{stage}_loss", total_loss)

        if accuracy:
            for i, (out, target, acc_metric) in enumerate(
                zip(outputs, targets, self.accuracies)
            ):
                acc = acc_metric(out, target)
                self.log(f"{stage}_acc_task{i}", acc)
            avg_acc = sum(
                [
                    acc_metric(out, target)
                    for acc_metric, out, target in zip(
                        self.accuracies, outputs, targets
                    )
                ]
            ) / len(outputs)
            self.log(f"{stage}_acc", avg_acc)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", accuracy=True)

    def test_step(self, batch, batch_idx):
        x, targets = batch
        y_pred = self(x)

        true_labels = [target.cpu().numpy() for target in targets]
        pred_labels = [np.argmax(pred.cpu().numpy(), axis=1) for pred in y_pred]

        self.test_step_outputs.append((true_labels, pred_labels))
        return true_labels, pred_labels

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        true_labels = list(zip(*[out[0] for out in outputs]))
        pred_labels = list(zip(*[out[1] for out in outputs]))

        for i, (y_true, y_pred) in enumerate(zip(true_labels, pred_labels)):
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_true, y_pred)
            conf_acc = np.trace(conf_matrix) / np.sum(conf_matrix)

            self.log(f"test_task_{i}_accuracy", accuracy)
            self.log(f"test_task_{i}_f1", f1)
            self.log(f"test_task_{i}_conf_acc", conf_acc)

            print(f"Task {i} Accuracy: {accuracy:.2f}")
            print(f"Task {i} F1 Score: {f1:.2f}")
            print(f"Task {i} Confusion Matrix:")
            print(conf_matrix)

        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
