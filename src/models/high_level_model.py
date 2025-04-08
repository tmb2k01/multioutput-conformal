import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models


class HighLevelModel(pl.LightningModule):
    def __init__(self, num_classes_list, learning_rate=1e-3):
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
                for num_classes in num_classes_list
            ]
        )

        # One accuracy metric per task
        self.accuracies = nn.ModuleList(
            [
                torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                for num_classes in num_classes_list
            ]
        )

        self.loss_fn = F.cross_entropy

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
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
