import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models


class MultiOutputModel(pl.LightningModule):
    def __init__(self, num_classes_task1=10, num_classes_task2=5, learning_rate=1e-3):
        super(MultiOutputModel, self).__init__()
        self.save_hyperparameters()

        # Load a pre-trained ResNet50 model (without the classification head)
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # remove final FC

        in_features = base_model.fc.in_features

        self.classifier1 = nn.Linear(in_features, num_classes_task1)
        self.classifier2 = nn.Linear(in_features, num_classes_task2)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        out1 = self.classifier1(features)  # logits for task 1
        out2 = self.classifier2(features)  # logits for task 2
        return out1, out2

    def shared_step(self, batch, stage):
        x, (y1, y2) = batch  # Expecting two label tensors
        out1, out2 = self(x)

        loss1 = self.loss_fn(out1, y1)
        loss2 = self.loss_fn(out2, y2)
        loss = loss1 + loss2

        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_loss_task1", loss1)
        self.log(f"{stage}_loss_task2", loss2)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
