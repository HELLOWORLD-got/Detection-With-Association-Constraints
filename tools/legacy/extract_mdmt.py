"""CRC-check and extract the downloaded archive without executing packaged code."""

import json
from pathlib import Path
import zipfile
from collections import Counter


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "archive" / "incomplete_hf"
    archive = root / "downloads" / "Multi-Drone-Multi-Object-Detection-and-Tracking.zip"
    out = root / "raw"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        for i, info in enumerate(infos):
            target = (out / info.filename).resolve()
            if not target.is_relative_to(out.resolve()) or info.filename.startswith("/"):
                raise RuntimeError(f"Unsafe archive path: {info.filename}")
            # Reading/extracting through ZipFile verifies each entry's CRC.
            z.extract(info, out)
            if i % 1000 == 0:
                print(f"Extracted {i}/{len(infos)} entries", flush=True)
        names = [x.filename for x in infos if not x.is_dir()]
    image_counts = Counter()
    sequences = {}
    for name in names:
        p = Path(name)
        if p.suffix.lower() == ".jpg":
            rel = p.parts[1:]
            image_counts["/".join(rel[:2])] += 1
            sequences.setdefault("/".join(rel[:3]), set()).add(p.stem)
    mismatches = []
    for seq, frames in sorted(sequences.items()):
        split, view, clip = seq.split("/")
        if view != "1":
            continue
        partner = f'{split}/2/{clip.rsplit("-", 1)[0]}-2'
        other = sequences.get(partner, set())
        if frames != other:
            mismatches.append(
                dict(
                    sequence=seq,
                    partner=partner,
                    view1_frames=len(frames),
                    view2_frames=len(other),
                    matched_frames=len(frames & other),
                    missing_in_view2=len(frames - other),
                    missing_in_view1=len(other - frames),
                )
            )
    report = dict(
        archive_crc="passed during extraction",
        total_files=len(names),
        images=sum(image_counts.values()),
        xml=sum(n.lower().endswith(".xml") for n in names),
        image_counts=dict(image_counts),
        pair_mismatches=mismatches,
        complete_official_dataset=False,
        note="Community mirror; fewer than the official 39,678 images. Do not use as complete training benchmark.",
    )
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"EXTRACTED {out}", flush=True)


if __name__ == "__main__":
    main()
