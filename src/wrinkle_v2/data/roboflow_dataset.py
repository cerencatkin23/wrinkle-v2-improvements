from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import apply_augmentations


class RoboflowWrinkleDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        input_size: int,
        augmentations: Callable | None = None,
    ) -> None:
        split = split.lower()
        if split == "val":
            split = "valid"
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = int(input_size)
        self.augmentations = augmentations

        self.images_dir = os.path.join(dataset_root, split, "images")
        self.masks_dir = os.path.join(dataset_root, split, "masks")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")

        self.items = self._collect_items()
        if not self.items:
            raise RuntimeError(f"No image/mask pairs found under {self.images_dir}")

    def _collect_items(self) -> List[Tuple[str, str]]:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        items: List[Tuple[str, str]] = []
        for name in sorted(os.listdir(self.images_dir)):
            if not name.lower().endswith(exts):
                continue
            stem = os.path.splitext(name)[0]
            mask_path = os.path.join(self.masks_dir, stem + ".png")
            if not os.path.exists(mask_path):
                continue
            img_path = os.path.join(self.images_dir, name)
            items.append((img_path, mask_path))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.items[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        if self.augmentations is not None:
            augmented = apply_augmentations(self.augmentations, image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return {"image": image_t, "mask": mask_t}
