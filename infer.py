"""Synchronized images -> query IDs/boxes JSON and annotated preview (no NMS)."""

import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from association.engine import build_model
from association.metrics import decode, xyxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="runs/inference")
    p.add_argument("--confidence", type=float)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = checkpoint["config"]
    device = torch.device(a.device)
    if len(a.images) != c["model"]["num_views"]:
        p.error("Number of images must equal configured num_views")
    originals = [Image.open(name).convert("RGB") for name in a.images]
    images = [
        TF.normalize(
            TF.to_tensor(TF.resize(im, [c["data"]["height"], c["data"]["width"]], antialias=True)),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        .unsqueeze(0)
        .to(device)
        for im in originals
    ]
    model = build_model(c, False).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        outputs = model(images, return_logits=True)
    pred = decode(
        outputs,
        a.confidence if a.confidence is not None else c["evaluation"]["confidence_threshold"],
    )[0]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for v, im in enumerate(originals):
        w, h = im.size
        boxes = xyxy(pred["boxes"][v]) * torch.tensor([w, h, w, h])
        draw = ImageDraw.Draw(im)
        detections = []
        for q in pred["present"][v].nonzero().flatten().tolist():
            box = boxes[q].tolist()
            color = ((q * 73 + 90) % 256, (q * 151 + 80) % 256, (q * 43 + 100) % 256)
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], box[1]), str(q), fill=color)
            detections.append(
                dict(
                    query_id=q,
                    label=int(pred["labels"][q]),
                    score=float(pred["scores"][v, q]),
                    xyxy=box,
                )
            )
        im.save(out / f"view_{v+1}.jpg")
        result.append(dict(image=a.images[v], detections=detections))
    (out / "predictions.json").write_text(
        json.dumps(
            dict(
                views=result,
                identity_scope="Query IDs associate these simultaneous views only; IDs are not tracked over time.",
            ),
            indent=2,
        )
    )
    print(out.resolve())


if __name__ == "__main__":
    main()
