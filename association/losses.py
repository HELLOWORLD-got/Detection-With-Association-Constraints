"""Paper Eqs. 20-25; labels 0..C-1, background C, boxes normalized cxcywh."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou


class HungarianLoss(nn.Module):
    def __init__(
        self, lambda_box=1.0, lambda_reg=1.0, lambda_conf=1.0, lambda_l1=1.0, lambda_giou=1.0
    ):
        super().__init__()
        self.weights = lambda_box, lambda_reg, lambda_conf, lambda_l1, lambda_giou

    def box_cost(self, pred, target):
        _, _, _, l1, giou = self.weights
        return l1 * torch.cdist(pred, target, p=1) + giou * (
            1
            - generalized_box_iou(
                box_convert(pred, "cxcywh", "xyxy"), box_convert(target, "cxcywh", "xyxy")
            )
        )

    def forward(self, outputs, targets, sync_distributed=False):
        use_logits = isinstance(outputs, dict)
        if use_logits:
            log_probs = outputs["pred_logits"].float().log_softmax(-1)
            boxes = outputs["pred_boxes"].float()
            scores = outputs["pred_visibility_logits"].float()
            probs = log_probs.exp()
        else:
            probs, boxes, scores = outputs
            boxes, scores = torch.stack(boxes, 1), torch.stack(scores, 1)
            log_probs = probs.clamp_min(1e-8).log()
        batch, views, queries, _ = boxes.shape
        if len(targets) != batch:
            raise ValueError("One target dictionary required per batch item")
        lb, lr, lc, _, _ = self.weights
        cls_losses, conf_losses = [], []
        regression = boxes.sum() * 0
        count = 0
        for b, target in enumerate(targets):
            labels = target["labels"].to(probs.device).long()
            gt = target["boxes"].to(boxes)
            visible = target["visible"].to(device=boxes.device, dtype=torch.bool)
            m = len(labels)
            if m > queries or gt.shape != (views, m, 4) or visible.shape != (views, m):
                raise ValueError("Target overflow or invalid boxes/visibility shapes")
            if m and (labels.min() < 0 or labels.max() >= probs.shape[-1] - 1):
                raise ValueError("Invalid foreground label")
            padded = torch.full(
                (queries,), probs.shape[-1] - 1, device=probs.device, dtype=torch.long
            )
            padded[:m] = labels
            cost = -log_probs[b, :, padded].detach().clone()
            for k in range(views):
                valid = visible[k].nonzero().flatten()
                if len(valid):
                    cost[:, valid] += lb * self.box_cost(boxes[b, k].detach(), gt[k, valid])
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row = torch.as_tensor(row, device=boxes.device)
            col = torch.as_tensor(col, device=boxes.device)
            cls_losses.append(F.nll_loss(log_probs[b, row], padded[col]))
            visibility_target = torch.zeros_like(scores[b])
            for k in range(views):
                real = col < m
                r, c = row[real], col[real]
                valid = visible[k, c]
                r, c = r[valid], c[valid]
                visibility_target[k, r] = 1
                if len(r):
                    regression = regression + self.box_cost(boxes[b, k, r], gt[k, c]).diag().sum()
                    count += len(r)
            conf_losses.append(
                (F.binary_cross_entropy_with_logits if use_logits else F.binary_cross_entropy)(
                    scores[b], visibility_target
                )
            )
        cls = torch.stack(cls_losses).mean()
        conf = torch.stack(conf_losses).mean()
        if sync_distributed:
            # DDP averages gradients: normalize by the global number of visible
            # boxes and compensate for the following world-size gradient average.
            normalizer = regression.new_tensor(float(count))
            dist.all_reduce(normalizer)
            reg = regression * dist.get_world_size() / normalizer.clamp_min(1)
        else:
            reg = regression / max(count, 1)
        return {
            "loss": cls + lr * reg + lc * conf,
            "classification": cls,
            "regression": reg,
            "confidence": conf,
        }
