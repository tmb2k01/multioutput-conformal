import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models


class HighLevelModel(pl.LightningModule):
    def __init__(self, num_classes_color=12, num_classes_type=11, learning_rate=1e-3):
        super(HighLevelModel, self).__init__()
        self.save_hyperparameters()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        in_features = base_model.fc.in_features

        self.classifier_color = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_color),
        )

        self.classifier_type = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_type),
        )

        self.loss_fn = F.cross_entropy
        self.accuracy_color = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes_color
        )
        self.accuracy_type = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes_type
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out_color = self.classifier_color(features)
        out_type = self.classifier_type(features)
        return out_color, out_type

    def shared_step(self, batch, stage, accuracy=False):
        x, (y_color, y_type) = batch
        out_color, out_type = self(x)

        loss_color = self.loss_fn(out_color, y_color)
        loss_type = self.loss_fn(out_type, y_type)
        loss = loss_color + loss_type

        self.log(f"{stage}_loss", loss)

        if accuracy:
            acc_color = self.accuracy_color(out_color, y_color)
            acc_type = self.accuracy_type(out_type, y_type)
            self.log(f"{stage}_acc_color", acc_color)
            self.log(f"{stage}_acc_type", acc_type)
            self.log(f"{stage}_acc", (acc_color + acc_type) / 2)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", accuracy=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
