import os

import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomVerticalFlip


def list_image_paths(root_dir):
    return [
        os.path.join(root_dir, filename) for filename in sorted(os.listdir(root_dir))
    ]


class HighLevelDataset(Dataset):
    def __init__(self, root_dir, task_num_classes, transform=None):
        img_dir = os.path.join(root_dir, "images")
        label_dir = os.path.join(root_dir, "labels")

        assert os.path.exists(root_dir), f"Root directory {root_dir} does not exist."
        assert os.path.exists(label_dir), f"Label directory {label_dir} does not exist."
        assert os.path.isdir(img_dir), f"Image directory {img_dir} is not a directory."

        self.img_paths = list_image_paths(img_dir)
        self.label_paths = list_image_paths(label_dir)
        self.task_num_classes = task_num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)

        label_path = self.label_paths[idx]
        with open(label_path, "r") as f:
            text = f.read().strip()
            labels = list(map(int, text.split()))

        if len(labels) != len(self.task_num_classes):
            raise ValueError(
                f"The number of labels does not match the number of tasks. Expected {len(self.task_num_classes)} labels, got {len(labels)}."
            )

        if self.transform:
            image = self.transform(image)

        # Return the image and a tuple of labels for each task
        return image, tuple(labels)


class HighLevelDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, task_num_classes, batch_size=32, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.num_classes_list = task_num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create datasets for different stages (train, val, test, calib)
        self.train_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "train"),
            task_num_classes=self.num_classes_list,
            transform=Compose([RandomVerticalFlip()]),
        )
        self.val_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "valid"),
            task_num_classes=self.num_classes_list,
        )
        self.test_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "test"),
            task_num_classes=self.num_classes_list,
        )
        self.calib_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "calib"),
            task_num_classes=self.num_classes_list,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def calib_dataloader(self):
        return DataLoader(
            self.calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
