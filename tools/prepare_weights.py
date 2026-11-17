"""Download standard torchvision ImageNet weights and verify filename hash prefix."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet50", "vit_b16", "all"], default="all")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1] / "weights"
    root.mkdir(exist_ok=True)
    records = {}
    for name, weight in [
        ("resnet50", ResNet50_Weights.DEFAULT),
        ("vit_b16", ViT_B_16_Weights.DEFAULT),
    ]:
        if a.model not in (name, "all"):
            continue
        filename = weight.url.rsplit("/", 1)[1]
        dest = root / filename
        if not dest.exists():
            temp = dest.with_suffix(".part")
            with urllib.request.urlopen(weight.url, timeout=120) as src, temp.open("wb") as out:
                while block := src.read(1024 * 1024):
                    out.write(block)
            temp.replace(dest)
        digest = hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()
        if not digest.startswith(filename.rsplit("-", 1)[1].split(".")[0]):
            raise ValueError(f"Corrupt weight file {dest}")
        records[name] = dict(url=weight.url, sha256=digest, size=dest.stat().st_size)
        print(name, records[name], flush=True)
    (root / "sources.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
