"""Frame-local cross-view metrics, never temporal tracking metrics.
MDA uses paper Eq.26 with one-to-one IoU>=.5 detection assignment.
UCR uses exact per-view presence, class, and strictly IoU>.6.
"""

from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import box_convert, box_iou


def xyxy(boxes):
    return box_convert(boxes.float(), "cxcywh", "xyxy").clamp(0, 1)


def maximum_valid_matching(valid, quality=None):
    if not valid.numel():
        return [], []
    # Maximize cardinality first; IoU breaks ties without sacrificing matches.
    value = valid.float() * (min(valid.shape) + 1)
    if quality is not None:
        value = value + quality.nan_to_num(0.0, posinf=0.0, neginf=0.0) * valid
    r, c = linear_sum_assignment(-value.cpu().numpy())
    good = valid[r, c].cpu().numpy()
    return r[good].tolist(), c[good].tolist()


def decode(outputs, confidence=0.5):
    if isinstance(outputs, dict):
        probs = outputs["pred_logits"].float().softmax(-1)
        boxes = outputs["pred_boxes"].unbind(1)
        scores = outputs["pred_visibility_logits"].float().sigmoid().unbind(1)
    else:
        probs, boxes, scores = outputs
    boxes = torch.stack(list(boxes), 1).detach().float().cpu()
    scores = torch.stack(list(scores), 1).detach().float().cpu()
    probs = probs.detach().float().cpu()
    category = probs[:, :, :-1].argmax(-1)
    foreground = probs.argmax(-1) != probs.shape[-1] - 1
    class_scores = probs[:, :, :-1].max(-1).values
    return [
        dict(
            boxes=boxes[b],
            labels=category[b],
            present=(scores[b] >= confidence) & foreground[b][None],
            scores=scores[b] * class_scores[b][None],
        )
        for b in range(len(probs))
    ]


class AssociationMetrics:
    def __init__(self, iou_threshold=0.5, ucr_threshold=0.6):
        self.iou_threshold = iou_threshold
        self.ucr_threshold = ucr_threshold
        self.sequence_mda = defaultdict(list)
        self.true = self.predicted = self.ground_truth = 0
        self.correct = self.instances = self.frames = 0

    def update(self, pred, target):
        boxes = target["boxes"].cpu()
        visible = target["visible"].cpu()
        labels = target["labels"].cpu()
        k, m = visible.shape
        maps = []
        all_ious = []
        for v in range(k):
            ious = box_iou(xyxy(pred["boxes"][v]), xyxy(boxes[v])).nan_to_num(
                0.0, posinf=0.0, neginf=0.0
            )
            all_ious.append(ious)
            allowed = (
                (ious >= self.iou_threshold) & pred["present"][v, :, None] & visible[v, None, :]
            )
            # Official MDMT association evaluation is category-agnostic.
            rows, cols = maximum_valid_matching(allowed, ious)
            maps.append(dict(zip(rows, cols)))
        for a, b in combinations(range(k), 2):
            linked = (pred["present"][a] & pred["present"][b]).nonzero().flatten().tolist()
            ga = int((visible[a] & visible[b]).sum())
            ta = sum(q in maps[a] and q in maps[b] and maps[a][q] == maps[b][q] for q in linked)
            ra = len(linked)
            fa = ra - ta
            ma = ga - ta
            denominator = ga + fa + ma
            # Empty GT & empty output is correct. Report this explicit convention.
            value = ta / denominator if denominator else 1.0
            self.sequence_mda[str(target["sequence"])].append(value)
            self.true += ta
            self.predicted += ra
            self.ground_truth += ga
        # Full-view assignment is independent of per-view IoU matching.
        valid = torch.ones((len(pred["labels"]), m), dtype=torch.bool)
        for v in range(k):
            same_presence = pred["present"][v, :, None] == visible[v, None, :]
            valid &= same_presence & (~visible[v, None, :] | (all_ious[v] > self.ucr_threshold))
        valid &= pred["labels"][:, None] == labels[None, :]
        rows, _ = maximum_valid_matching(valid)
        self.correct += len(rows)
        self.instances += m
        self.frames += 1

    def compute(self):
        frame_values = [v for values in self.sequence_mda.values() for v in values]
        precision = self.true / self.predicted if self.predicted else 0.0
        recall = self.true / self.ground_truth if self.ground_truth else 0.0
        per_sequence = {s: float(np.mean(v)) for s, v in self.sequence_mda.items()}
        return dict(
            MDA=float(np.mean(list(per_sequence.values()))) if per_sequence else 0.0,
            MDA_frame_mean=float(np.mean(frame_values)) if frame_values else 0.0,
            UCR=self.correct / self.instances if self.instances else 0.0,
            association_precision=precision,
            association_recall=recall,
            association_f1=(
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            ),
            true_associations=self.true,
            predicted_associations=self.predicted,
            ground_truth_associations=self.ground_truth,
            correct_instances=self.correct,
            total_instances=self.instances,
            frames=self.frames,
            MDA_per_sequence=per_sequence,
        )


class DetectionMetrics:
    """COCO bbox AP over view images, standard maxDets=100, not a custom AP."""

    def __init__(self, num_classes):
        self.classes = num_classes
        self.images = []
        self.annotations = []
        self.detections = []

    def update(self, pred, target):
        for v, size in enumerate(target["sizes"]):
            image_id = len(self.images) + 1
            w, h = size
            self.images.append(dict(id=image_id, width=w, height=h))
            scale = torch.tensor([w, h, w, h])
            truth = xyxy(target["boxes"][v]) * scale
            for j in target["visible"][v].nonzero().flatten().tolist():
                x1, y1, x2, y2 = truth[j].tolist()
                self.annotations.append(
                    dict(
                        id=len(self.annotations) + 1,
                        image_id=image_id,
                        category_id=int(target["labels"][j]) + 1,
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        area=(x2 - x1) * (y2 - y1),
                        iscrowd=0,
                    )
                )
            predicted = xyxy(pred["boxes"][v]) * scale
            for q, score in enumerate(pred["scores"][v].tolist()):
                if score < 0.001:
                    continue
                x1, y1, x2, y2 = predicted[q].tolist()
                self.detections.append(
                    dict(
                        image_id=image_id,
                        category_id=int(pred["labels"][q]) + 1,
                        bbox=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        score=score,
                    )
                )

    def compute(self):
        import contextlib, io
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self.annotations:
            return dict(mAP=None, AP50=None, AP75=None)
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = dict(
                images=self.images,
                annotations=self.annotations,
                categories=[dict(id=i + 1, name=str(i)) for i in range(self.classes)],
                info={},
            )
            gt.createIndex()
            if self.detections:
                dt = gt.loadRes(self.detections)
            else:
                dt = COCO()
                dt.dataset = dict(
                    images=self.images, annotations=[], categories=gt.dataset["categories"]
                )
                dt.createIndex()
            ev = COCOeval(gt, dt, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return dict(mAP=float(ev.stats[0]), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]))
