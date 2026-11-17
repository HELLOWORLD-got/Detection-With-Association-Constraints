"""One entry point: raw MDMT -> audited train/val/test manifests."""

import argparse
from pathlib import Path
from tools.prepare_data import prepare, fetch_official_test_gt, prepare_official_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/MDMT", help="Original MDMT directory")
    parser.add_argument(
        "--output", default="data/processed/MDMT", help="Generated manifests and audit"
    )
    parser.add_argument(
        "--verify-images", action="store_true", help="Check every JPEG, not just metadata"
    )
    parser.add_argument("--class-mode", choices=["agnostic", "native"], default="agnostic")
    parser.add_argument("--xml-only", action="store_true", help="Skip official TXT test manifest")
    args = parser.parse_args()
    prepare(args.root, args.output, args.class_mode, args.verify_images)
    if not args.xml_only:
        if args.class_mode != "agnostic":
            parser.error("Official TXT test labels require class-mode agnostic")
        output = Path(args.output)
        fetch_official_test_gt(output / "official_test")
        report = prepare_official_test(
            output / "test.jsonl", output / "official_test", output / "test_official.jsonl"
        )
        print("Official test manifest:", report)


if __name__ == "__main__":
    main()
