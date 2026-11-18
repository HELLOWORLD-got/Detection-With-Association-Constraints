"""Read-only checkpoint diagnostics on evenly spaced train/val samples."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision.ops import box_iou
from tqdm import tqdm
from association.engine import build_model, criterion_from, make_loader, normalize_config_paths
from association.metrics import decode, xyxy


def quantiles(values):
    if not values:
        return None
    x = torch.cat([v.detach().cpu().float().reshape(-1) for v in values])
    if not x.numel():
        return None
    return dict(zip(("min", "p50", "p90", "max"),
                    torch.quantile(x, torch.tensor([0., .5, .9, 1.])).tolist()))


@torch.no_grad()
def diagnose(model, config, split, device, limit):
    loader = make_loader(config, split, batch_size=1)
    # Spread samples across the declared split rather than inspect one clip only.
    ds = loader.dataset
    indices = torch.linspace(0, len(ds) - 1, min(limit, len(ds))).long().tolist()
    ds.offsets = [ds.offsets[i] for i in indices]
    criterion = criterion_from(config)
    values, counts, losses = defaultdict(list), defaultdict(int), defaultdict(float)
    frames = []
    for images, targets in tqdm(loader, desc=f"Diagnose {split}", file=sys.stderr):
        out = model([im.to(device) for im in images], return_logits=True)
        if not all(torch.isfinite(v).all() for v in out.values()):
            raise RuntimeError(f"Non-finite model output on {split}")
        for key, loss in criterion(out, targets).items():
            losses[key] += loss.item()
        probs = out["pred_logits"].softmax(-1).cpu()[0]
        vis = out["pred_visibility_logits"].sigmoid().cpu()[0]
        boxes = out["pred_boxes"].cpu()[0]
        foreground = probs.argmax(-1) != probs.shape[-1] - 1
        fg_prob = probs[:, :-1].max(-1).values
        pred = decode(out, config["evaluation"]["confidence_threshold"])[0]
        target = targets[0]
        counts["samples"] += 1
        counts["foreground_queries"] += int(foreground.sum())
        counts["queries"] += len(foreground)
        values["foreground_probability"].append(fg_prob)
        values["visibility_probability"].append(vis)
        values["combined_ap_score"].append(vis * fg_prob[None])
        values["query_box_std"].append(boxes.std(dim=1, unbiased=False))
        values["predicted_width"].append(boxes[..., 2])
        values["predicted_height"].append(boxes[..., 3])
        frame = dict(sequence=target["sequence"], frame=target["frame"], views=[])
        for v in range(len(images)):
            gt = target["boxes"][v, target["visible"][v]]
            ious = box_iou(xyxy(boxes[v]), xyxy(gt))
            best = ious.max(0).values if len(gt) else torch.empty(0)
            kept = pred["present"][v]
            kept_best = ious[kept].max(0).values if kept.any() and len(gt) else torch.zeros(len(gt))
            counts["gt_boxes"] += len(gt)
            counts["mda_kept_boxes"] += int(kept.sum())
            counts["ap_candidates_score_ge_0.001"] += int((pred["scores"][v] >= .001).sum())
            counts["gt_with_raw_iou_ge_0.5"] += int((best >= .5).sum())
            counts["gt_with_kept_iou_ge_0.5"] += int((kept_best >= .5).sum())
            values["best_raw_iou_per_gt"].append(best)
            values["gt_width"].append(gt[:, 2])
            values["gt_height"].append(gt[:, 3])
            frame["views"].append(dict(gt=len(gt), mda_kept=int(kept.sum()),
                                        raw_iou_ge_0_5=int((best >= .5).sum())))
        frames.append(frame)
    n = counts["samples"]
    return dict(
        counts=dict(counts),
        losses={key: value / n for key, value in losses.items()},
        quantiles={key: quantiles(value) for key, value in values.items()},
        frames=frames,
        note="Raw IoU coverage ignores scores, class and one-to-one matching; it is not AP or MDA.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=16, help="Evenly spaced samples PER split")
    p.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    p.add_argument("--output", default="runs/diagnosis.json")
    args = p.parse_args()
    if args.limit < 1:
        p.error("--limit must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = normalize_config_paths(checkpoint["config"])
    training_config = dict(config["training"])
    config["training"]["workers"] = 0
    device = torch.device(args.device)
    model = build_model(config, pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), epoch=checkpoint["epoch"],
                  model=config["model"], training=training_config,
                  evaluation=config["evaluation"], splits={})
    for split in args.splits:
        report["splits"][split] = diagnose(model, config, split, device, args.limit)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    summary = {**report, "splits": {key: {k: v for k, v in value.items() if k != "frames"}
                                      for key, value in report["splits"].items()}}
    print(json.dumps(summary, indent=2))
    print(f"Full report: {path}")


if __name__ == "__main__":
    main()
