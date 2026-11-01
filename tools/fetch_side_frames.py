"""Grab one frame per side-angle camera for Nutrition5k dishes, without downloading whole videos.

Nutrition5k filmed every dish with 4 Raspberry Pi cameras (A-D) at alternating 30 and 60 degree angles
(dataset README); each raw .h264 video is ~30-45 MB. The first few MB already hold the first frames, so an
HTTP range request for the head of the file is enough for one still per camera. A frame is kept only when
the next frame also decodes (a cut-off last frame comes back with smeared streaks); otherwise the head is doubled. These stills are weighed
dishes photographed from an angle with a different camera from the overhead images the portion model was
trained on: an out-of-distribution test that looks much more like a phone photo.

Data licence (Nutrition5k README): "We release all Nutrition5k data under the Creative Commons V4.0 license."

    python tools/fetch_side_frames.py --ids data/raw/nutrition5k/dish_ids/splits/depth_test_ids.txt \
        --out data/external/n5k_side --limit 20
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # a cut-off stream ends mid-frame: expected, not worth a warning
import cv2  # noqa: E402
import numpy as np  # noqa: E402

URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles/{dish}/camera_{cam}.h264"
FIELDS = ["dish_id", "camera", "file", "frame", "head_bytes", "width", "height"]


def decode_head(data: bytes, frame: int) -> np.ndarray | None:
    """Frame `frame` of a raw h264 head, but only if frame `frame + 1` also decodes. The last frame of a cut-off
    stream comes back "successfully" with its missing part filled by the decoder (smeared streaks), so a frame
    is only known to be complete once the next one has started."""
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        cap = cv2.VideoCapture(str(tmp))
        keep, i = None, 0
        ok, img = cap.read()
        while ok:
            if i == frame:
                keep = img
            elif i == frame + 1:
                cap.release()
                return keep
            i += 1
            ok, img = cap.read()
        cap.release()
        return None
    finally:
        tmp.unlink(missing_ok=True)


def fetch_frame(dish: str, cam: str, out: Path, head_bytes: int, frame: int, max_side: int | None,
                retries: int = 2) -> dict | None:
    target = out / f"{dish}_{cam}.jpg"
    if target.exists():
        im = cv2.imread(str(target))
        return {"dish_id": dish, "camera": cam, "file": target.name, "frame": frame, "head_bytes": "",
                "width": im.shape[1], "height": im.shape[0]}
    size = head_bytes
    for _ in range(retries + 1):
        req = urllib.request.Request(URL.format(dish=dish, cam=cam), headers={"Range": f"bytes=0-{size - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 416):
                return None   # this camera was not recorded for this dish
            raise
        img = decode_head(data, frame)
        if img is not None or len(data) < size:   # complete frame, or the whole video is shorter than the head
            break
        size *= 2
    if img is None:
        return None
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"dish_id": dish, "camera": cam, "file": target.name, "frame": frame, "head_bytes": size,
            "width": img.shape[1], "height": img.shape[0]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", type=Path, required=True, help="text file of dish ids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--head-bytes", type=int, default=1_600_000, help="first request size; doubled on retry")
    ap.add_argument("--frame", type=int, default=3, help="frame index to keep (kept only if complete)")
    ap.add_argument("--max-side", type=int, default=960, help="downscale stills (plate views are 320 px anyway)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    ids = [l.strip() for l in args.ids.read_text(encoding="utf-8").splitlines() if l.strip()][:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(d, c) for d in ids for c in args.cameras]
    rows, failed = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_frame, d, c, args.out, args.head_bytes, args.frame, args.max_side) for d, c in jobs]
        for n, fut in enumerate(futures, 1):
            try:
                row = fut.result()
            except Exception as e:   # one bad download must not stop the batch; it is counted and reported
                failed += 1
                print(f"failed {jobs[n - 1]}: {e}")
                continue
            if row:
                rows.append(row)
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} requests, {len(rows)} frames", flush=True)
    with open(args.out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["dish_id"], r["camera"])))
    print(f"{len(rows)} frames from {len({r['dish_id'] for r in rows})} dishes; {failed} failed requests")


if __name__ == "__main__":
    main()
