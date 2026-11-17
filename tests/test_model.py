import unittest
import torch
from torch import nn
from association.models import InstanceBasedAssociation, get_resnet_backbone, get_vit_backbone
from association.losses import HungarianLoss


class ModelTests(unittest.TestCase):
    def test_multiview_backward_and_masks(self):
        torch.manual_seed(7)
        for views in (2, 3):
            model = InstanceBasedAssociation(
                nn.Conv2d(3, 32, 3, stride=8), hidden_dim=32, num_queries=4, num_views=views
            )
            outputs = model([torch.randn(2, 3, 32, 32) for _ in range(views)])
            self.assertEqual(outputs[0].shape, (2, 4, 2))
            targets = [
                dict(
                    labels=torch.tensor([0]),
                    boxes=torch.full((views, 1, 4), 0.3),
                    visible=torch.tensor([[1]] + [[0]] * (views - 1), dtype=torch.bool),
                ),
                dict(
                    labels=torch.empty(0, dtype=torch.long),
                    boxes=torch.empty(views, 0, 4),
                    visible=torch.empty(views, 0, dtype=torch.bool),
                ),
            ]
            criterion = HungarianLoss()
            losses = criterion(outputs, targets)
            changed = [dict(t) for t in targets]
            changed[0]["boxes"] = targets[0]["boxes"].clone()
            changed[0]["boxes"][1:] = 0.9
            self.assertTrue(torch.allclose(losses["loss"], criterion(outputs, changed)["loss"]))
            losses["loss"].backward()
            self.assertTrue(torch.isfinite(losses["loss"]))
            self.assertIsNotNone(model.mvoq.object_queries.grad)
            self.assertTrue(
                all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            )

    def test_rectangular_vit_features(self):
        backbone = get_vit_backbone(False, hidden_dim=32).eval()
        with torch.no_grad():
            features = backbone(torch.randn(1, 3, 49, 65))
        self.assertEqual(features.shape, (1, 32, 4, 5))

    def test_resnet_forward(self):
        model = InstanceBasedAssociation(get_resnet_backbone(False)).eval()
        with torch.no_grad():
            outputs = model([torch.randn(1, 3, 64, 64) for _ in range(2)])
        self.assertEqual(outputs[1][0].shape, (1, 25, 4))


if __name__ == "__main__":
    unittest.main()
