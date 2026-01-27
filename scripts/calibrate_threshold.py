import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.data.augmentations import get_val_augmentations
from wrinkle_v2.data.roboflow_dataset import RoboflowWrinkleDataset
from wrinkle_v2.models.unet import build_unet
from wrinkle_v2.utils import get_device, load_config


def dice_from_probs(probs: np.ndarray, targets: np.ndarray, threshold: float, smooth: float = 1.0) -> float:
    preds = (probs > threshold).astype(np.float32)
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (preds * targets).sum(axis=1)
    union = preds.sum(axis=1) + targets.sum(axis=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(np.mean(dice))


def collect_probs(model, loader, device) -> tuple:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="calib", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].cpu().numpy()
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(masks)
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def main(cfg: Dict, checkpoint_path: str, thresholds: List[float]):
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
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    val_ds = RoboflowWrinkleDataset(cfg["dataset_root"], "valid", cfg["input_size"], get_val_augmentations())
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    probs, targets = collect_probs(model, val_loader, device)
    best_thr = 0.5
    best_dice = -1.0
    scores = {}
    for thr in thresholds:
        d = dice_from_probs(probs, targets, thr)
        scores[f"{thr:.2f}"] = d
        if d > best_dice:
            best_dice = d
            best_thr = thr

    result = {
        "best_threshold": float(best_thr),
        "best_dice": float(best_dice),
        "scores": scores,
    }

    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, "threshold_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/wrinkle_unet_best.pt")
    parser.add_argument("--min", type=float, default=0.2, dest="min_thr")
    parser.add_argument("--max", type=float, default=0.8, dest="max_thr")
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    thresholds = []
    t = args.min_thr
    while t <= args.max_thr + 1e-6:
        thresholds.append(round(t, 4))
        t += args.step

    cfg = load_config(args.config)
    main(cfg, args.checkpoint, thresholds)
