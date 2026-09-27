"""Convert TartanAir image_left/*.png to image_left_gray/*_left.jpg, the layout reference/rl_vo expects.

    .venv/bin/python scripts/tartan_to_gray.py [data/TartanAir] [--quality 95]

Skips images that are already converted. The authors' JPEG quality is unknown; we use OpenCV's default (95).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def convert(args):
    src, dst, quality = args
    if dst.exists():
        return 0
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not read {src}")
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(ROOT / "data/TartanAir"))
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    jobs = []
    for traj in sorted(Path(args.root).glob("*/*/P*")):
        src_dir, dst_dir = traj / "image_left", traj / "image_left_gray"
        if not src_dir.is_dir():
            continue
        dst_dir.mkdir(exist_ok=True)
        for png in sorted(src_dir.glob("*_left.png")):
            jobs.append((png, dst_dir / (png.stem + ".jpg"), args.quality))

    with ProcessPoolExecutor() as ex:
        n = sum(ex.map(convert, jobs, chunksize=64))
    print(f"converted {n} / {len(jobs)} images")


if __name__ == "__main__":
    main()
