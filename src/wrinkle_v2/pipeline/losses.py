import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)


def _morphological_gradient(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.bce = nn.BCELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.clamp(0.0, 1.0)
        target_edges = _morphological_gradient(targets, self.kernel_size)
        pred_edges = _morphological_gradient(probs, self.kernel_size)
        return self.bce(pred_edges, target_edges)


class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.3,
        boundary_weight: float = 0.2,
        boundary_kernel: int = 3,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss(kernel_size=boundary_kernel)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.bce_weight * self.bce(logits, targets)
            + self.boundary_weight * self.boundary(logits, targets)
        )
