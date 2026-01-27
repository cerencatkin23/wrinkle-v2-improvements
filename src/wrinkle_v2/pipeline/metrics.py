from typing import Iterable

import torch
import torch.nn.functional as F


def dice_coeff(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> float:
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> float:
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    total = probs.sum(dim=1) + targets.sum(dim=1)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def _boundary_map(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_f1(logits: torch.Tensor, targets: torch.Tensor, kernel_size: int = 3, smooth: float = 1.0) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = _boundary_map(preds, kernel_size)
    targets = _boundary_map(targets, kernel_size)

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    precision = (tp + smooth) / (preds.sum(dim=1) + smooth)
    recall = (tp + smooth) / (targets.sum(dim=1) + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    return f1.mean().item()


def stability_score(scores: Iterable[float]) -> float:
    vals = list(scores)
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    max_delta = max(vals) - min(vals)
    return float(1.0 / (1.0 + std + max_delta))
