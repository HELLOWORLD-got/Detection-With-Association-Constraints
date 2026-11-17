"""Evaluate a checkpoint on a declared split; optionally export frame-local IDs."""

import argparse
import json
from pathlib import Path
import torch
from association.engine import make_loader, normalize_config_paths
from association.engine import build_model, criterion_from, evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--manifest")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--export-predictions")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = normalize_config_paths(checkpoint["config"])
    c["training"]["workers"] = 0
    c["training"]["batch_size"] = a.batch_size
    if a.manifest:
        c["data"][a.split] = a.manifest
    loader = make_loader(c, a.split, a.limit)
    device = torch.device(a.device)
    model = build_model(c, False).to(device)
    model.load_state_dict(checkpoint["model"])
    result = evaluate_model(
        model, loader, criterion_from(c), c, device, predictions_path=a.export_predictions
    )
    result.update(
        checkpoint=str(Path(a.checkpoint).resolve()),
        epoch=checkpoint["epoch"],
        split=a.split,
        limited=a.limit is not None,
        configuration=c,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "configuration"}, indent=2))


if __name__ == "__main__":
    main()
