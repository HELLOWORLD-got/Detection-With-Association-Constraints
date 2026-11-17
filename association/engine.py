import json
from pathlib import Path
import random
import torch
from torch import nn
from .models import InstanceBasedAssociation, get_resnet_backbone, get_vit_backbone
from .losses import HungarianLoss
from .data import MultiViewDataset, collate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .metrics import AssociationMetrics, DetectionMetrics, decode, xyxy


def normalize_config_paths(config):
    """Keep existing state-dict checkpoints usable after the directory cleanup."""
    for split in ("train", "val", "test"):
        value = config.get("data", {}).get(split)
        prefix = "data/mdmt_complete/"
        if isinstance(value, str) and value.startswith(prefix):
            config["data"][split] = "data/processed/MDMT/" + value[len(prefix) :]
    return config


def load_config(path):
    return normalize_config_paths(json.loads(Path(path).read_text()))


def seed_all(seed, deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def build_model(config, pretrained=None):
    m = config["model"]
    choice = m.get("backbone", "resnet50")
    use_pretrained = m.get("pretrained", True) if pretrained is None else pretrained
    if choice == "resnet50":
        backbone = get_resnet_backbone(use_pretrained, m["hidden_dim"], m.get("pretrained_weights"))
    elif choice == "vit_b16":
        backbone = get_vit_backbone(use_pretrained, m["hidden_dim"], m.get("pretrained_weights"))
    else:
        raise ValueError(f"Unknown backbone {choice}")
    model = InstanceBasedAssociation(
        backbone,
        hidden_dim=m["hidden_dim"],
        num_queries=m["num_queries"],
        num_classes=m["num_classes"],
        num_views=m["num_views"],
        num_layers=m["num_layers"],
    )
    # Frozen BN statistics are stable for per-view single-GPU small batches.
    # This is an explicit training choice; affine parameters remain trainable.
    return model


def freeze_bn_stats(model):
    for layer in model.backbone.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def criterion_from(config):
    return HungarianLoss(**config["loss"])


def outputs_float(outputs):
    if isinstance(outputs, dict):
        return {k: v.float() for k, v in outputs.items()}
    return outputs[0].float(), [b.float() for b in outputs[1]], [s.float() for s in outputs[2]]


def make_loader(
    config, split, limit=None, shuffle=False, augment=False, world_size=1, rank=0, batch_size=None
):
    d, m, t = config["data"], config["model"], config["training"]
    dataset = MultiViewDataset(
        d[split],
        (d["height"], d["width"]),
        m["num_views"],
        augment,
        m["num_queries"],
        limit,
        m["num_classes"],
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=t["seed"],
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else t["batch_size"],
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=t["workers"],
        # NCCL (and some Gloo transports) are not fork-safe. Use fresh
        # workers for both training and rank-0 validation under DDP.
        multiprocessing_context=(
            "spawn" if t["workers"] > 0 and torch.distributed.is_initialized() else None
        ),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_model(
    model, loader, criterion, config, device, limit_batches=None, predictions_path=None
):
    model.eval()
    association = AssociationMetrics(**config["evaluation"]["association"])
    detection = DetectionMetrics(config["model"]["num_classes"])
    losses = {}
    count = 0
    export = {} if predictions_path else None
    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        outputs = outputs_float(model([im.to(device) for im in images], return_logits=True))
        ls = criterion(outputs, targets)
        for key, value in ls.items():
            losses[key] = losses.get(key, 0.0) + value.item() * len(targets)
        predictions = decode(outputs, config["evaluation"]["confidence_threshold"])
        for pred, target in zip(predictions, targets):
            association.update(pred, target)
            detection.update(pred, target)
            if export is not None:
                for v, size in enumerate(target["sizes"]):
                    key = f"{target['sequence']}-{v+1}"
                    frame = str(target["frame"])
                    scale = torch.tensor([size[0], size[1], size[0], size[1]])
                    boxes = xyxy(pred["boxes"][v]) * scale
                    export.setdefault(key, {})[frame] = [
                        [q, *boxes[q].tolist()]
                        for q in pred["present"][v].nonzero().flatten().tolist()
                    ]
        count += len(targets)
    if not count:
        raise ValueError("No evaluation samples")
    result = association.compute()
    result.update(detection.compute())
    result["losses"] = {k: v / count for k, v in losses.items()}
    result["samples"] = count
    if predictions_path:
        out = Path(predictions_path)
        out.mkdir(parents=True, exist_ok=True)
        for name, frames in export.items():
            # Preserve actual XML frame keys; never replace them with enumerate indices.
            ordered = {k: frames[k] for k in sorted(frames, key=int)}
            (out / (name + ".json")).write_text(json.dumps(ordered))
    return result
