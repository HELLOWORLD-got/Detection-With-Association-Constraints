import unittest
import torch
from torch import nn
from association.models import (
    InstanceBasedAssociation, MultiViewObjectQuery, get_resnet_backbone, get_vit_backbone,
)
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
            self.assertIsNotNone(model.mvoq.attention.in_proj_weight.grad)
            self.assertIsNotNone(model.mvoq.object_queries.grad)
            self.assertTrue(
                all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            )

    def test_author_mvoq_matches_repeated_attention_in_instance_order(self):
        torch.manual_seed(7)
        module = MultiViewObjectQuery(hidden_dim=32, num_queries=3, mode="author")
        instances = torch.randn(2, 3, 32, requires_grad=True)
        view = torch.randn(2, 32, 2, 3, requires_grad=True)

        # Author's expanded calculation, with repetitions grouped by instance.
        tokens = view.flatten(2).transpose(1, 2)
        locations = tokens.shape[1]
        queries = instances.repeat_interleave(locations, dim=1)
        keys_values = tokens.repeat(1, instances.shape[1], 1)
        ref, _ = module.attention(queries, keys_values, keys_values)
        ref = ref.reshape(2, 3, locations, 32).mean(dim=2)
        expected_boxes = module.box_head(ref).sigmoid()
        expected_visibility = module.confidence_head(ref).squeeze(-1)

        boxes, visibility = module(instances, [view], return_logits=True)
        torch.testing.assert_close(boxes[0], expected_boxes)
        torch.testing.assert_close(visibility[0], expected_visibility)
        self.assertNotIn("object_queries", module.state_dict())
        self.assertFalse(torch.allclose(boxes[0][:, 0], boxes[0][:, 1]))
        actual_grad = torch.autograd.grad(boxes[0].sum() + visibility[0].sum(), (instances, view))
        expected_grad = torch.autograd.grad(
            expected_boxes.sum() + expected_visibility.sum(), (instances, view)
        )
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)

    def test_rectangular_vit_features(self):
        backbone = get_vit_backbone(False, hidden_dim=32).eval()
        with torch.no_grad():
            features = backbone(torch.randn(1, 3, 49, 65))
        self.assertEqual(features.shape, (1, 32, 4, 5))

    def test_triplet_attention_matches_explicit_instance_conditioned_keys(self):
        torch.manual_seed(19)
        module = MultiViewObjectQuery(hidden_dim=32, num_queries=3, mode="triplet")
        instances = torch.randn(2, 3, 32, requires_grad=True)
        view = torch.randn(2, 32, 2, 3, requires_grad=True)
        tokens = view.flatten(2).transpose(1, 2)
        query = module.query_projection(module.object_queries[0]).reshape(3, 8, 4)
        gate = (1 + module.instance_projection(instances).tanh()).reshape(2, 3, 8, 4)
        key, value = module.view_projection(tokens).chunk(2, dim=-1)
        keys = gate[:, :, None] * key.reshape(2, 6, 8, 4)[:, None]
        logits = torch.einsum("nhd,bnshd->bnhs", query, keys) / 2
        features = torch.einsum("bnhs,bshd->bnhd", logits.softmax(-1), value.reshape(2, 6, 8, 4))
        features = module.output_projection(features.flatten(-2))

        boxes, visibility = module(instances, [view], return_logits=True)
        torch.testing.assert_close(boxes[0], module.box_head(features).sigmoid())
        torch.testing.assert_close(visibility[0], module.confidence_head(features).squeeze(-1))
        (boxes[0].sum() + visibility[0].sum()).backward()
        for grad in (instances.grad, view.grad, module.object_queries.grad[0],
                     module.instance_projection.weight.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_resnet_forward(self):
        model = InstanceBasedAssociation(get_resnet_backbone(False)).eval()
        with torch.no_grad():
            outputs = model([torch.randn(1, 3, 64, 64) for _ in range(2)])
        self.assertEqual(outputs[1][0].shape, (1, 25, 4))


if __name__ == "__main__":
    unittest.main()
