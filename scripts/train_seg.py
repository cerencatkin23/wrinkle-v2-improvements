import argparse
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.data.augmentations import get_train_augmentations, get_val_augmentations
from wrinkle_v2.data.roboflow_dataset import RoboflowWrinkleDataset
from wrinkle_v2.models.unet import build_unet
from wrinkle_v2.pipeline.losses import CombinedSegLoss, DiceBCELoss
from wrinkle_v2.pipeline.metrics import dice_coeff, iou_score
from wrinkle_v2.utils import get_device, load_config, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


def evaluate(model, loader, device):
    model.eval()
    dices = []
    ious = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            dices.append(dice_coeff(logits, masks))
            ious.append(iou_score(logits, masks))
    return float(sum(dices) / max(1, len(dices))), float(sum(ious) / max(1, len(ious)))


def main(cfg: Dict):
    set_seed(cfg["seed"])
    device = get_device()

    train_ds = RoboflowWrinkleDataset(cfg["dataset_root"], "train", cfg["input_size"], get_train_augmentations())
    val_ds = RoboflowWrinkleDataset(cfg["dataset_root"], "valid", cfg["input_size"], get_val_augmentations())

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model_cfg = cfg["model"]
    model = build_unet(model_cfg["encoder"], model_cfg["in_channels"], model_cfg["out_channels"], model_cfg["pretrained"]).to(device)

    boundary_weight = float(cfg["loss"].get("boundary_weight", 0.0))
    if boundary_weight > 0:
        criterion = CombinedSegLoss(
            dice_weight=cfg["loss"]["dice_weight"],
            bce_weight=cfg["loss"]["bce_weight"],
            boundary_weight=boundary_weight,
            boundary_kernel=int(cfg["loss"].get("boundary_kernel", 3)),
        )
    else:
        criterion = DiceBCELoss(cfg["loss"]["dice_weight"], cfg["loss"]["bce_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_dice = 0.0
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, cfg["train"]["best_name"])
    export_path = os.path.join(save_dir, cfg["train"]["export_name"])
    last_path = os.path.join(save_dir, "wrinkle_unet_last.pt")

    start_epoch = 0
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_dice = float(checkpoint.get("best_dice", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming from {last_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if (epoch + 1) % cfg["train"]["val_interval"] == 0:
            val_dice, val_iou = evaluate(model, val_loader, device)
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)
                model.eval()
                example = torch.randn(1, 3, cfg["input_size"], cfg["input_size"], device=device)
                traced = torch.jit.trace(model, example)
                traced.save(export_path)
            print(f"epoch {epoch+1}: loss={loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")
        else:
            print(f"epoch {epoch+1}: loss={loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
            },
            last_path,
        )

    print(f"best dice: {best_dice:.4f} saved to {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
