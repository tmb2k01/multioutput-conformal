import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models


class MultiOutputModel(pl.LightningModule):
    def __init__(self, num_classes_color=12, num_classes_type=11, learning_rate=1e-3):
        super(MultiOutputModel, self).__init__()
        self.save_hyperparameters(learning_rate=learning_rate)

        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        in_features = base_model.fc.in_features

        self.classifier_color = nn.Linear(in_features, num_classes_color)
        self.classifier_type = nn.Linear(in_features, num_classes_type)

        self.loss_fn = nn.CrossEntropyLoss()

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
        self.log(f"{stage}_loss_color", loss_color)
        self.log(f"{stage}_loss_type", loss_type)

        if accuracy:
            acc_color = (out_color.argmax(dim=1) == y_color).float().mean()
            acc_type = (out_type.argmax(dim=1) == y_type).float().mean()
            self.log(f"{stage}_acc_color", acc_color)
            self.log(f"{stage}_acc_type", acc_type)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val", accuracy=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
