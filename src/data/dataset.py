import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


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

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    LABEL_EXTENSIONS = {".txt"}

    def __init__(self, root_dir, task_num_classes, transform=None) -> None:
        img_dir = os.path.join(root_dir, "images")
        label_dir = os.path.join(root_dir, "labels")

        assert os.path.exists(root_dir), f"Root directory {root_dir} does not exist."
        assert os.path.isdir(
            img_dir
        ), f"Image directory {img_dir} does not exist or is not a directory."
        assert os.path.isdir(
            label_dir
        ), f"Label directory {label_dir} does not exist or is not a directory."

        image_map = self._list_files_by_stem(img_dir, self.IMAGE_EXTENSIONS)
        label_map = self._list_files_by_stem(label_dir, self.LABEL_EXTENSIONS)

        missing_labels = sorted(image_map.keys() - label_map.keys())
        missing_images = sorted(label_map.keys() - image_map.keys())

        if missing_labels or missing_images:
            raise ValueError(
                "Image/label file mismatch detected.\n"
                f"Missing label files for image stems: {missing_labels[:10]}\n"
                f"Missing image files for label stems: {missing_images[:10]}"
            )

        common_stems = sorted(image_map.keys())
        self.samples = [(image_map[stem], label_map[stem]) for stem in common_stems]

        if not self.samples:
            raise ValueError(f"No valid image/label pairs found under {root_dir}")

        self.task_num_classes = task_num_classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads and returns an image and its corresponding multi-task labels.

        Returns:
            Tuple[Tensor, Tensor]: Image tensor and tensor of task labels.
        """
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)

        with open(label_path, encoding="utf-8") as f:
            text = f.read().strip()
            labels = list(map(int, text.split()))

        if len(labels) != len(self.task_num_classes):
            raise ValueError(
                f"Expected {len(self.task_num_classes)} labels, but got {len(labels)} in {label_path}."
            )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _list_files_by_stem(root_dir, allowed_extensions):
        """
        Build a mapping:
            file stem -> full path

        Example:
            cat_001.jpg -> {"cat_001": "/path/.../cat_001.jpg"}
        """
        files_by_stem = {}

        for filename in os.listdir(root_dir):
            full_path = os.path.join(root_dir, filename)
            if not os.path.isfile(full_path):
                continue

            stem, ext = os.path.splitext(filename)
            if ext.lower() not in allowed_extensions:
                continue

            if stem in files_by_stem:
                raise ValueError(
                    f"Duplicate file stem '{stem}' found in {root_dir}. "
                    "Pairing would be ambiguous."
                )

            files_by_stem[stem] = full_path

        return files_by_stem