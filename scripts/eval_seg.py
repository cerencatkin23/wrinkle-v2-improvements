import argparse
import json
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.data.augmentations import get_val_augmentations
from wrinkle_v2.data.roboflow_dataset import RoboflowWrinkleDataset
from wrinkle_v2.models.unet import build_unet
from wrinkle_v2.pipeline.metrics import dice_coeff, iou_score
from wrinkle_v2.utils import get_device, load_config


def evaluate_split(model, loader, device):
    model.eval()
    dices = []
    ious = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            dices.append(dice_coeff(logits, masks))
            ious.append(iou_score(logits, masks))
    return float(sum(dices) / max(1, len(dices))), float(sum(ious) / max(1, len(ious)))


def main(cfg: Dict, checkpoint_path: str):
    device = get_device()
    model_cfg = cfg["model"]
    model = build_unet(model_cfg["encoder"], model_cfg["in_channels"], model_cfg["out_channels"], model_cfg["pretrained"]).to(device)

    if os.path.isdir(checkpoint_path):
        candidate = os.path.join(checkpoint_path, "checkpoints", "wrinkle_unet_best.pt")
        if os.path.exists(candidate):
            checkpoint_path = candidate
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), "..", checkpoint_path)
    if os.path.isdir(checkpoint_path):
        candidate = os.path.join(checkpoint_path, "checkpoints", "wrinkle_unet_best.pt")
        if os.path.exists(candidate):
            checkpoint_path = candidate
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    val_ds = RoboflowWrinkleDataset(cfg["dataset_root"], "valid", cfg["input_size"], get_val_augmentations())
    test_ds = RoboflowWrinkleDataset(cfg["dataset_root"], "test", cfg["input_size"], get_val_augmentations())
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    val_dice, val_iou = evaluate_split(model, val_loader, device)
    test_dice, test_iou = evaluate_split(model, test_loader, device)

    metrics = {
        "valid": {"dice": val_dice, "iou": val_iou},
        "test": {"dice": test_dice, "iou": test_iou},
    }

    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/wrinkle_unet_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.checkpoint)
