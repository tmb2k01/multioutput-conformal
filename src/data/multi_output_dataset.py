import os

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomVerticalFlip


def list_image_paths(root_dir):
    """
    Lists and sorts full paths to all files in a directory.

    Args:
        root_dir (str): Path to the directory.

    Returns:
        List[str]: Sorted list of file paths.
    """
    return [
        os.path.join(root_dir, filename)
        for filename in sorted(os.listdir(root_dir))
        if os.path.isfile(os.path.join(root_dir, filename))
    ]


class MultiOutputDataset(Dataset):
    """
    A custom Dataset for multi-task classification with one label file per image.

    Each image has an associated text file containing space-separated integers,
    where each integer corresponds to the class index for a specific task.

    Args:
        root_dir (str): Path to the dataset root containing 'images' and 'labels' subfolders.
        task_num_classes (List[int]): Number of classes for each task.
        transform (callable, optional): Optional transform to apply to images.
    """

    def __init__(self, root_dir, task_num_classes, transform=None):
        img_dir = os.path.join(root_dir, "images")
        label_dir = os.path.join(root_dir, "labels")

        assert os.path.exists(root_dir), f"Root directory {root_dir} does not exist."
        assert os.path.isdir(
            img_dir
        ), f"Image directory {img_dir} does not exist or is not a directory."
        assert os.path.isdir(
            label_dir
        ), f"Label directory {label_dir} does not exist or is not a directory."

        self.img_paths = list_image_paths(img_dir)
        self.label_paths = list_image_paths(label_dir)

        if len(self.img_paths) != len(self.label_paths):
            raise ValueError("The number of images and label files does not match.")

        self.task_num_classes = task_num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Loads and returns an image and its corresponding multi-task labels.

        Returns:
            Tuple[Tensor, Tuple[int]]: Image tensor and a tuple of task labels.
        """
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)

        with open(label_path, "r") as f:
            text = f.read().strip()
            labels = list(map(int, text.split()))

        if len(labels) != len(self.task_num_classes):
            raise ValueError(
                f"Expected {len(self.task_num_classes)} labels, but got {len(labels)} in {label_path}."
            )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)


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

    def __init__(self, root_dir, task_num_classes, batch_size=32, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.task_num_classes = task_num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Sets up the dataset splits for training, validation, testing, and calibration.

        Args:
            stage (str, optional): Can be used to specify setup stage. Not used here.
        """
        self.train_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, "train"),
            task_num_classes=self.task_num_classes,
            transform=Compose([RandomVerticalFlip()]),  # Simple data augmentation
        )
        self.val_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, "valid"),
            task_num_classes=self.task_num_classes,
        )
        self.test_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, "test"),
            task_num_classes=self.task_num_classes,
        )
        self.calib_dataset = MultiOutputDataset(
            root_dir=os.path.join(self.root_dir, "calib"),
            task_num_classes=self.task_num_classes,
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
