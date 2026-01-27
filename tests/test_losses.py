import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.pipeline.losses import BoundaryLoss, CombinedSegLoss


class TestBoundaryLoss(unittest.TestCase):
    def test_boundary_loss_deterministic(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
        loss_fn = BoundaryLoss(kernel_size=3)
        loss1 = loss_fn(logits, targets)
        loss2 = loss_fn(logits, targets)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)

    def test_boundary_loss_grad(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(1, 1, 16, 16) > 0.5).float()
        loss_fn = BoundaryLoss(kernel_size=3)
        loss = loss_fn(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_combined_loss_scalar(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = (torch.rand(2, 1, 16, 16) > 0.5).float()
        loss_fn = CombinedSegLoss(dice_weight=0.5, bce_weight=0.3, boundary_weight=0.2)
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
