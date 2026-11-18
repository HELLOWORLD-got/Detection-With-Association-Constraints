"""Fit one real, unaugmented training pair before committing to a long experiment.
This is a diagnostic of memorization, not a validation accuracy measurement.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from association.engine import (load_config, seed_all, make_loader, build_model,
                                freeze_bn_stats, criterion_from, evaluate_model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/resnet50.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--index", type=int, default=334, help="Zero-based index in the train manifest")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--output", default="runs/overfit_diagnostic")
    args = p.parse_args()
    if args.steps < 1 or args.lr <= 0:
        p.error("steps and lr must be positive")
    config = load_config(args.config)
    config["training"]["batch_size"] = 1
    config["training"]["workers"] = 0
    config["data"]["augment"] = False
    seed_all(config["training"]["seed"])
    loader = make_loader(config, "train")
    if not 0 <= args.index < len(loader.dataset):
        p.error("index is outside train manifest")
    ds = loader.dataset
    ds.offsets = [ds.offsets[args.index]]
    images, targets = next(iter(loader))
    device = torch.device(args.device)
    images = [im.to(device) for im in images]
    model = build_model(config).to(device)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    criterion = criterion_from(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=config["training"]["weight_decay"])
    # FP32, no augmentation or AMP: remove variables from this learning check.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    began = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        freeze_bn_stats(model)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(images, return_logits=True), targets)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("Non-finite overfit loss")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad_norm"],
                                      error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            result = evaluate_model(model, loader, criterion, config, device)
            record = dict(step=step, **result)
            history.append(record)
            print(json.dumps(record), flush=True)
    report = dict(arguments=vars(args), config=config, sequence=targets[0]["sequence"],
                  frame=targets[0]["frame"], instances=len(targets[0]["labels"]),
                  elapsed_seconds=time.monotonic()-began, history=history,
                  note="One training pair memorization only; not evidence of generalization or paper reproduction.")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(dict(model=model.state_dict(), config=config, epoch=0, diagnostic_step=args.steps,
                    diagnostic_only=True), out / "diagnostic.pt")
    print(f"Saved diagnostic results to {out}", flush=True)


if __name__ == "__main__":
    main()
