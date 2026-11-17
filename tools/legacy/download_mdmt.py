"""Download the public MDMT HF mirror with resume, then verify its LFS SHA256."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / "data" / "archive" / "incomplete_hf"
DEST = ROOT / "downloads"
REPO = "https://huggingface.co/datasets/mrhwl123/MDMT"
REVISION = "3b6511a4472420f877eacd04edad0a1384bfe7e5"
NAME = "Multi-Drone-Multi-Object-Detection-and-Tracking.zip"
SIZE = 16706405892
SHA256 = "189c105f6480d45b0eb92b4e7e6d4e8ca1d54b6af2d91fcd507150eb5cf90cd4"
CHUNK = 4 * 1024 * 1024


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    part = DEST / (NAME + ".part")
    complete = DEST / NAME
    if complete.exists():
        path = complete
    else:
        # Cache-buster avoids stale signed CDN URLs from intermediary caches.
        url = f"{REPO}/resolve/{REVISION}/{NAME}?download=true&t={int(time.time())}"
        state_path = DEST / "download_state.json"
        done = set(
            json.loads(state_path.read_text()) if state_path.exists() and part.exists() else []
        )
        fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, SIZE)

        def fetch(i):
            start, end = i * CHUNK, min((i + 1) * CHUNK, SIZE) - 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(
                        url,
                        headers={
                            "Range": f"bytes={start}-{end}",
                            "User-Agent": "MDMT-research-download/1.0",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=90) as response:
                        expected = f"bytes {start}-{end}/{SIZE}"
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range") != expected
                        ):
                            raise RuntimeError(
                                f'Invalid range response: {response.status}, {response.headers.get("Content-Range")}'
                            )
                        offset = start
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if offset + len(block) > end + 1:
                                raise RuntimeError("Oversized range response")
                            written = 0
                            while written < len(block):
                                written += os.pwrite(fd, block[written:], offset + written)
                            offset += len(block)
                        if offset != end + 1:
                            raise RuntimeError(
                                f"Incomplete range response: {offset-start}/{end-start+1} bytes"
                            )
                    return i
                except Exception as exc:
                    print(f"chunk={i} retry={attempt+1}: {exc}", flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(min(2**attempt, 20))

        started = time.monotonic()
        last_report = 0
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
                futures = [
                    pool.submit(fetch, i)
                    for i in range((SIZE + CHUNK - 1) // CHUNK)
                    if i not in done
                ]
                for future in concurrent.futures.as_completed(futures):
                    done.add(future.result())
                    os.fsync(fd)
                    tmp = state_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(sorted(done)))
                    tmp.replace(state_path)
                    if time.monotonic() - last_report > 15:
                        progress = sum(min(CHUNK, SIZE - i * CHUNK) for i in done)
                        print(
                            f"{progress/1e9:.2f}/{SIZE/1e9:.2f} GB ({progress/SIZE:.1%}), elapsed={time.monotonic()-started:.0f}s",
                            flush=True,
                        )
                        last_report = time.monotonic()
        finally:
            os.close(fd)
        path = part
    print("Verifying SHA256...", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise RuntimeError(f"Integrity check failed: {digest.hexdigest()}")
    if path != complete:
        path.rename(complete)
    (ROOT / "source.json").write_text(
        json.dumps(
            dict(
                source=REPO, revision=REVISION, file=NAME, size=SIZE, sha256=SHA256, verified=True
            ),
            indent=2,
        )
    )
    print(f"COMPLETE {complete}", flush=True)


if __name__ == "__main__":
    main()
