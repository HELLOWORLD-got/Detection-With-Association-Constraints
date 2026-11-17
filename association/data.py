"""MDMT CVAT XML parsing. XML frame=0 corresponds to the first sorted JPEG, irrespective of filename.
Track IDs are shared across cameras in the benchmark evaluation (official TXT IDs
are XML IDs + 1). Occluded boxes are valid detections, not absent instances.
"""

from collections import defaultdict
import xml.etree.ElementTree as ET


def read_tracks(path, width, height, class_mode="agnostic"):
    frames = defaultdict(dict)
    labels = {}
    stats = defaultdict(int)
    classes = {"person": 0, "car": 1, "bicycle": 2, "bus": 3}
    for track in ET.parse(path).getroot().findall("track"):
        identity = int(track.attrib["id"])
        name = track.attrib["label"]
        if name not in classes:
            raise ValueError(f"Unknown category {name} in {path}")
        label = 0 if class_mode == "agnostic" else classes[name]
        labels[identity] = name
        stats["tracks_" + name] += 1
        for box in track.findall("box"):
            stats["xml_boxes"] += 1
            if int(box.get("outside", "0")):
                stats["outside"] += 1
                continue
            x1, y1, x2, y2 = [float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
            if not all(__import__("math").isfinite(x) for x in [x1, y1, x2, y2]):
                raise ValueError(f"Non-finite coordinates in {path}")
            clipped = [
                max(0.0, min(width, x1)),
                max(0.0, min(height, y1)),
                max(0.0, min(width, x2)),
                max(0.0, min(height, y2)),
            ]
            stats["clipped"] += int(clipped != [x1, y1, x2, y2])
            x1, y1, x2, y2 = clipped
            if not all(__import__("math").isfinite(x) for x in clipped) or x2 <= x1 or y2 <= y1:
                stats["invalid_boxes"] += 1
                continue
            frame = int(box.attrib["frame"])
            if identity in frames[frame]:
                raise ValueError(f"Duplicate ID {identity} frame {frame} in {path}")
            frames[frame][identity] = dict(
                label=label,
                box=[
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
                occluded=bool(int(box.get("occluded", "0"))),
            )
    return frames, labels, dict(stats)


"""Portable synchronized JSON/JSONL data; scene IDs preserved through augmentation."""

import json
from pathlib import Path
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, ColorJitter


class MultiViewDataset(Dataset):
    def __init__(
        self,
        manifest,
        size=(405, 720),
        num_views=2,
        augment=False,
        num_queries=None,
        limit=None,
        num_classes=None,
    ):
        self.path = Path(manifest).resolve()
        self.size, self.num_views, self.augment = tuple(size), num_views, augment
        self.jitter = ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.offsets = []
        self.samples = None
        if self.path.suffix == ".jsonl":
            with self.path.open("rb") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append(pos)
        else:
            self.samples = json.loads(self.path.read_text())
            self.offsets = list(range(len(self.samples)))
        if limit is not None:
            self.offsets = self.offsets[:limit]
        if not self.offsets:
            raise ValueError("Empty manifest")
        self.max_instances = 0
        for i in range(len(self)):
            s = self.record(i)
            self.max_instances = max(self.max_instances, len(s["labels"]))
            if num_classes is not None and any(x < 0 or x >= num_classes for x in s["labels"]):
                raise ValueError("Manifest categories incompatible with model")
        if num_queries is not None and self.max_instances > num_queries:
            raise ValueError(
                f"{self.path.name}: maximum {self.max_instances} global instances > {num_queries} queries. "
                "No annotations were truncated. Use the full-MDMT configuration."
            )

    def __len__(self):
        return len(self.offsets)

    def record(self, index):
        if self.samples is not None:
            return self.samples[self.offsets[index]]
        with self.path.open("rb") as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())

    def __getitem__(self, index):
        s = self.record(index)
        if len(s["images"]) != self.num_views:
            raise ValueError("Invalid view count")
        m = len(s["labels"])
        boxes = torch.tensor(s["boxes"], dtype=torch.float32).reshape(self.num_views, m, 4)
        visible = torch.tensor(s["visible"], dtype=torch.bool).reshape(self.num_views, m)
        if not torch.isfinite(boxes).all() or ((boxes[visible] < 0) | (boxes[visible] > 1)).any():
            raise ValueError("Boxes must be finite normalized cxcywh")
        if (boxes[visible][:, 2:] <= 0).any():
            raise ValueError("Visible boxes need positive size")
        # One normalized crop/flip for the entire synchronized sample.
        crop = (0.0, 0.0, 1.0, 1.0)
        if self.augment and random.random() < 0.5:
            scale = random.uniform(0.8, 1.0)
            crop = (random.uniform(0, 1 - scale), random.uniform(0, 1 - scale), scale, scale)
        flip = self.augment and random.random() < 0.5
        images = []
        for v, name in enumerate(s["images"]):
            with Image.open(self.path.parent / name) as source:
                image = source.convert("RGB")
            w, h = image.size
            left, top, cw, ch = crop
            l, t = int(left * w), int(top * h)
            rw, rh = max(1, int(cw * w)), max(1, int(ch * h))
            # Use actual integer crop fractions so geometry matches pixels exactly.
            lo = torch.tensor([l / w, t / h])
            span = torch.tensor([rw / w, rh / h])
            lower = ((boxes[v, :, :2] - boxes[v, :, 2:] / 2) - lo) / span
            upper = ((boxes[v, :, :2] + boxes[v, :, 2:] / 2) - lo) / span
            lower, upper = lower.clamp(0, 1), upper.clamp(0, 1)
            visible[v] &= (upper > lower).all(-1)
            boxes[v] = torch.cat([(lower + upper) / 2, upper - lower], -1)
            image = TF.crop(image, t, l, rh, rw)
            if flip:
                image = TF.hflip(image)
                boxes[v, :, 0] = 1 - boxes[v, :, 0]
            if self.augment:
                image = self.jitter(image)
            image = TF.to_tensor(TF.resize(image, self.size, antialias=True))
            images.append(TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        keep = visible.any(0)
        boxes[:, ~visible.any(0)] = 0
        boxes[~visible] = 0
        target = dict(
            labels=torch.tensor(s["labels"], dtype=torch.long)[keep],
            boxes=boxes[:, keep],
            visible=visible[:, keep],
            ids=torch.tensor(s.get("ids", list(range(m))), dtype=torch.long)[keep],
            sequence=s.get("sequence", "sample"),
            frame=s.get("frame", index),
            sizes=s.get("sizes", [[self.size[1], self.size[0]]] * self.num_views),
            images=s["images"],
        )
        return images, target


def collate(samples):
    images, targets = zip(*samples)
    return [torch.stack(view) for view in zip(*images)], list(targets)
