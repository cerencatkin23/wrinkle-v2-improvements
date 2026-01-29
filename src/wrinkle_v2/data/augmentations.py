from __future__ import annotations

from typing import Any, Dict

import albumentations as A


def get_train_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, border_mode=0, value=0, mask_value=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.0), p=0.08),
        ]
    )


def get_val_augmentations() -> A.Compose:
    return A.Compose([])


def apply_augmentations(aug: A.Compose, image: Any, mask: Any) -> Dict[str, Any]:
    if aug is None:
        return {"image": image, "mask": mask}
    return aug(image=image, mask=mask)
