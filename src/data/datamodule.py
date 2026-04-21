import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomVerticalFlip, Resize

from data.dataset import MultiOutputDataset


class MultiOutputDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling multi-task datasets.

    Assumes the dataset directory contains subfolders: 'train', 'valid', 'test', and 'calib',
    each with 'images' and 'labels' inside.

    Args:
        root_dir (str): Root directory containing dataset splits.
        task_num_classes (List[int]): Number of classes for each task.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of subprocesses for data loading.
    """

    def __init__(self, root_dir, task_num_classes, batch_size=32, num_workers=4, split_idx=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.task_num_classes = task_num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iter = split_idx

        self.train_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, "train"),
            task_num_classes=self.task_num_classes,
            transform=Compose(
                [RandomVerticalFlip(), Resize((256, 256))]
            ),  # Simple data augmentation
        )
        self.val_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, "valid"),
            task_num_classes=self.task_num_classes,
            transform=Resize((256, 256)),
        )

        self.test_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, f"test{f'_{self.iter}'if self.iter is not None else ''}"),
            task_num_classes=self.task_num_classes,
            transform=Resize((256, 256)),
        )
        self.calib_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, f"calib{f'_{self.iter}'if self.iter is not None else ''}"),
            task_num_classes=self.task_num_classes,
            transform=Resize((256, 256)),
        )

        self.datasets = {
            "train": self.train_dataset,
            "valid": self.val_dataset,
            "test": self.test_dataset,
            "calib": self.calib_dataset,
        }

    def train_dataloader(self):
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Returns the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def calib_dataloader(self):
        """Returns the calibration DataLoader."""
        return DataLoader(
            self.calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
