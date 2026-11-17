"""MDMT preparation stages used by the project's prepare.py entry point."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from PIL import Image
from association.data import read_tracks

ROOT = Path(__file__).resolve().parents[1]
REV = "551f90d998087ea2d02df75700e3c7739c4ecbe1"
CLIPS = [26, 31, 34, 48, 52, 55, 56, 57, 59, 61, 62, 68, 71, 73]


def prepare(root, output, class_mode="agnostic", verify_images=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        root=str(root),
        class_mode=class_mode,
        frame_rule="XML frame = zero-based index in numerically sorted JPEG list (official supplement_MIA.py)",
        identity_rule="XML track ID shared across views; official TXT uses XML ID + 1",
        splits={},
        label_conflicts=[],
        annotations_sha256={},
        excluded_unannotated_trailing_frames=[],
    )
    split_clips = {}
    max_instances = 0
    for split in ("train", "val", "test"):
        dirs = [root / split / str(v) for v in (1, 2)]
        clips = [{p.name.rsplit("-", 1)[0]: p for p in d.iterdir() if p.is_dir()} for d in dirs]
        if clips[0].keys() != clips[1].keys():
            raise ValueError(
                f"{split}: missing paired sequences {clips[0].keys() ^ clips[1].keys()}"
            )
        split_clips[split] = set(clips[0])
        totals, hist = Counter(), Counter()
        temp = output / (split + ".jsonl.tmp")
        with temp.open("w") as stream:
            for clip in sorted(clips[0], key=int):
                images = [
                    {int(p.stem) - 1: p for p in clips[v][clip].glob("*.jpg") if p.stem.isdigit()}
                    for v in range(2)
                ]
                if images[0].keys() != images[1].keys() or not images[0]:
                    raise ValueError(f"{split}/{clip}: unequal or empty synchronized frames")
                dimensions = []
                for v in range(2):
                    with Image.open(next(iter(images[v].values()))) as im:
                        dimensions.append(im.size)
                    if verify_images:
                        for p in images[v].values():
                            with Image.open(p) as im:
                                if im.size != dimensions[v]:
                                    raise ValueError(f"Changing image dimensions: {p}")
                                im.verify()
                parsed = []
                for v in range(2):
                    xml = root / "new_xml" / str(v + 1) / f"{clip}-{v+1}.xml"
                    report["annotations_sha256"][str(xml.relative_to(root))] = hashlib.sha256(
                        xml.read_bytes()
                    ).hexdigest()
                    f, lab, stat = read_tracks(xml, *dimensions[v], class_mode=class_mode)
                    parsed.append((f, lab))
                    totals.update(stat)
                for identity in parsed[0][1].keys() & parsed[1][1].keys():
                    a, b = parsed[0][1][identity], parsed[1][1][identity]
                    if a != b:
                        report["label_conflicts"].append([clip, identity, a, b])
                for frame, image_number in enumerate(sorted(images[0])):
                    if frame > max(max(parsed[v][0], default=-1) for v in range(2)):
                        report["excluded_unannotated_trailing_frames"].append(
                            [split, clip, frame, image_number + 1]
                        )
                        continue
                    views = [parsed[v][0].get(frame, {}) for v in range(2)]
                    ids = sorted(views[0].keys() | views[1].keys())
                    labels = []
                    for identity in ids:
                        category = {view[identity]["label"] for view in views if identity in view}
                        if len(category) != 1:
                            raise ValueError(
                                f"Category conflict for shared ID {clip}/{frame}/{identity}; use agnostic mode or corrected labels"
                            )
                        labels.append(category.pop())
                    sample = dict(
                        sequence=clip,
                        frame=frame,
                        image_number=image_number + 1,
                        ids=ids,
                        labels=labels,
                        images=[os.path.relpath(images[v][image_number], output) for v in range(2)],
                        sizes=[list(x) for x in dimensions],
                        boxes=[
                            [
                                views[v][i]["box"] if i in views[v] else [0.0, 0.0, 0.0, 0.0]
                                for i in ids
                            ]
                            for v in range(2)
                        ],
                        visible=[[i in views[v] for i in ids] for v in range(2)],
                        occluded=[
                            [views[v][i]["occluded"] if i in views[v] else False for i in ids]
                            for v in range(2)
                        ],
                    )
                    stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    hist[len(ids)] += 1
                    totals["synchronized_frames"] += 1
                    totals["visible_boxes_in_images"] += sum(len(x) for x in views)
                    max_instances = max(max_instances, len(ids))
                totals["clips"] += 1
        temp.replace(output / (split + ".jsonl"))
        report["splits"][split] = dict(
            totals,
            images=2 * totals["synchronized_frames"],
            max_instances=max(hist, default=0),
            frames_above_25=sum(n for m, n in hist.items() if m > 25),
            instance_histogram=dict(sorted(hist.items())),
        )
        print(
            split,
            {k: v for k, v in report["splits"][split].items() if k != "instance_histogram"},
            flush=True,
        )
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if split_clips[a] & split_clips[b]:
            raise ValueError(f"Sequence leakage: {a}/{b}")
    report["sequence_leakage"] = False
    report["max_instances"] = max_instances
    report["total_images"] = sum(x["images"] for x in report["splits"].values()) + 2 * len(
        report["excluded_unannotated_trailing_frames"]
    )
    report["usable_images"] = sum(x["images"] for x in report["splits"].values())
    report["official_expected_images"] = 39678
    report["matches_official_image_count"] = report["total_images"] == 39678
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return report


def prepare_official_test(manifest, ground_truth, output):
    manifest, ground_truth, output = Path(manifest), Path(ground_truth), Path(output)
    if manifest.parent.resolve() != output.parent.resolve():
        raise ValueError("Keep manifests in the same directory for relative image paths")
    cache = {}
    kept = dropped = 0
    boxes = 0
    with output.with_suffix(".tmp").open("w") as stream:
        for line in manifest.open():
            s = json.loads(line)
            seq = s["sequence"]
            frame = s["frame"]
            if seq not in cache:
                views = []
                for v in (1, 2):
                    frames = defaultdict(dict)
                    for row in (ground_truth / f"{seq}-{v}.txt").read_text().splitlines():
                        x = row.split(",")
                        f = int(x[0]) - 1
                        identity = int(x[1]) - 1
                        left, top, w, h = map(float, x[2:6])
                        width, height = s["sizes"][v - 1]
                        x1, x2 = max(0.0, left), min(width, left + w)
                        y1, y2 = max(0.0, top), min(height, top + h)
                        if x2 <= x1 or y2 <= y1:
                            raise ValueError("Degenerate official box")
                        frames[f][identity] = [
                            (x1 + x2) / 2 / width,
                            (y1 + y2) / 2 / height,
                            (x2 - x1) / width,
                            (y2 - y1) / height,
                        ]
                    views.append(frames)
                cache[seq] = views
            views = cache[seq]
            if frame not in views[0] and frame not in views[1]:
                dropped += 1
                continue
            rows = [v.get(frame, {}) for v in views]
            ids = sorted(rows[0].keys() | rows[1].keys())
            if not set(ids) <= set(s["ids"]):
                raise ValueError("Official IDs do not agree with XML IDs + 1")
            s.update(
                ids=ids,
                labels=[0] * len(ids),
                visible=[[i in view for i in ids] for view in rows],
                boxes=[[view.get(i, [0.0, 0.0, 0.0, 0.0]) for i in ids] for view in rows],
                occluded=[[False] * len(ids) for _ in rows],
                annotation_source="official MDMT test TXT",
            )
            stream.write(json.dumps(s, separators=(",", ":")) + "\n")
            kept += 1
            boxes += sum(len(v) for v in rows)
    output.with_suffix(".tmp").replace(output)
    r = dict(
        samples=kept,
        images=2 * kept,
        boxes=boxes,
        unannotated_trailing_samples_excluded=dropped,
        categories="agnostic",
        source="Pinned official MDMT demo/eval/test TXT labels; frame and ID minus one",
    )
    output.with_suffix(".audit.json").write_text(json.dumps(r, indent=2))
    return r


def fetch_official_test_gt(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        url = f"https://raw.githubusercontent.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking/{REV}/demo/eval/test/{name}"
        p = out / name
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        return name, dict(url=url, sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    with ThreadPoolExecutor(6) as pool:
        records = dict(pool.map(fetch, [f"{clip}-{view}.txt" for clip in CLIPS for view in (1, 2)]))
    (out / "sources.json").write_text(json.dumps(records, indent=2))
    print("Official test files:", len(records))
