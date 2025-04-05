import os

import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def list_image_paths(root_dir):
    img_paths = []
    for filename in os.listdir(root_dir):
        img_path = os.path.join(root_dir, filename)
        img_paths.append(img_path)
    return img_paths


class HighLevelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        img_dir = os.path.join(root_dir, "images")
        label_dir = os.path.join(root_dir, "labels")

        assert os.path.exists(root_dir), f"Root directory {root_dir} does not exist."
        assert os.path.exists(label_dir), f"Label directory {label_dir} does not exist."
        assert os.path.isdir(img_dir), f"Image directory {img_dir} is not a directory."

        self.img_paths = list_image_paths(img_dir)
        self.label_path = list_image_paths(label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)

        label_path = self.label_path[idx]
        with open(label_path, "r") as f:
            text = f.read().strip()
            labels = (text[0], text[1])

        if self.transform:
            image = self.transform(image)

        return image, labels


class HighLevelDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = None
        # self.transform = transforms.Compose([transforms.Resize((128, 128))])

    def setup(self, stage=None):
        self.train_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "train"), transform=self.transform
        )
        self.val_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "val"), transform=self.transform
        )
        self.test_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "test"), transform=self.transform
        )
        self.calib_dataset = HighLevelDataset(
            root_dir=os.path.join(self.root_dir, "calib"), transform=self.transform
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
