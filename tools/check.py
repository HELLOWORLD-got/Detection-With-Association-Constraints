"""Read-only environment check, with optional real-data oracle evaluation."""

import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torchvision
from PIL import Image, ImageDraw
from association.data import MultiViewDataset
from association.engine import load_config
from association.metrics import AssociationMetrics, DetectionMetrics, xyxy
from tools.score_official_mda import score


def check_environment(config_path):
    c = load_config(config_path)
    info = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        cuda_available=torch.cuda.is_available(),
        torch_cuda=torch.version.cuda,
    )
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, vram_gb=round(prop.total_memory / 1024**3, 2))
    for split in ("train", "val", "test"):
        ds = MultiViewDataset(
            c["data"][split],
            num_queries=c["model"]["num_queries"],
            num_classes=c["model"]["num_classes"],
        )
        info[split] = dict(samples=len(ds), maximum_instances=ds.max_instances)
    weight = c["model"].get("pretrained_weights")
    if c["model"]["pretrained"] and weight and not Path(weight).exists():
        raise FileNotFoundError(weight)
    return info


def check_oracle():
    root = Path("runs/oracle_predictions")
    root.mkdir(parents=True, exist_ok=True)
    ds = MultiViewDataset("data/processed/MDMT/test_official.jsonl")
    seen = set()
    metrics = AssociationMetrics()
    detection = DetectionMetrics(1)
    for i in range(len(ds)):
        s = ds.record(i)
        if s["sequence"] in seen:
            continue
        seen.add(s["sequence"])
        m = len(s["labels"])
        t = dict(
            boxes=torch.tensor(s["boxes"]).reshape(2, m, 4),
            visible=torch.tensor(s["visible"]).reshape(2, m),
            labels=torch.tensor(s["labels"]),
            sequence=s["sequence"],
            sizes=s["sizes"],
        )
        pred = dict(
            boxes=t["boxes"], present=t["visible"], labels=t["labels"], scores=t["visible"].float()
        )
        metrics.update(pred, t)
        detection.update(pred, t)
        for v in (0, 1):
            w, h = s["sizes"][v]
            b = xyxy(t["boxes"][v]) * torch.tensor([w, h, w, h])
            rows = [
                [s["ids"][j] + 1, *b[j].tolist()]
                for j in t["visible"][v].nonzero().flatten().tolist()
            ]
            (root / f"{s['sequence']}-{v+1}.json").write_text(json.dumps({str(s["frame"]): rows}))
        if s["sequence"] == "52":
            preview = Image.new("RGB", (960, 270))
            for v in (0, 1):
                im = Image.open(ds.path.parent / s["images"][v]).convert("RGB")
                draw = ImageDraw.Draw(im)
                b = xyxy(t["boxes"][v]) * torch.tensor([im.width, im.height, im.width, im.height])
                for j in t["visible"][v].nonzero().flatten().tolist():
                    identity = s["ids"][j]
                    color = (
                        (identity * 73 + 90) % 256,
                        (identity * 151 + 80) % 256,
                        (identity * 43 + 100) % 256,
                    )
                    draw.rectangle(b[j].tolist(), outline=color, width=3)
                    draw.text(tuple(b[j, :2].tolist()), str(identity), fill=color)
                preview.paste(im.resize((480, 270)), (480 * v, 0))
            preview.save("runs/data_alignment_preview.jpg")
    r = metrics.compute()
    r.update(detection.compute())
    r["official"] = score(
        root, "data/processed/MDMT/official_test", "reference/mdmt_official/mango_eval.py"
    )
    r["note"] = "Ground-truth predictions, one real frame per test clip. Not model performance."
    Path("runs/oracle_check.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    assert r["MDA"] == 1 and r["UCR"] == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/resnet50.json")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Also check evaluation using ground-truth predictions (not model accuracy)",
    )
    args = parser.parse_args()
    print(json.dumps(check_environment(args.config), indent=2))
    if args.oracle:
        check_oracle()
