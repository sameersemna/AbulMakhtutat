"""Create a fixed, reproducible train/holdout manifest split for ketos training.

ketos `-t/--training-data` and `-e/--evaluation-data` take a *manifest file*
containing one image path per line (see `kraken/ketos/util.py::_validate_manifests`).
This writes both manifests so every future run trains on the same lines and is
evaluated on the same, genuinely held-out lines (no leakage, no random split).

Usage:
    .venv/bin/python make_holdout.py --data annotation_nodiac/images

    .venv/bin/ketos -d cuda:0 train -f path -o models/run_x \
        -t train.txt -e holdout.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Write train.txt / holdout.txt manifests for ketos.")
    ap.add_argument("--data", type=Path, default=Path("annotation_nodiac/images"), help="Directory of image+gt pairs.")
    ap.add_argument("--holdout-fraction", type=float, default=0.1, help="Fraction reserved for evaluation.")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    ap.add_argument("--train-out", type=Path, default=Path("train.txt"))
    ap.add_argument("--holdout-out", type=Path, default=Path("holdout.txt"))
    args = ap.parse_args()

    images = sorted(args.data.glob("*.png"))
    if not images:
        print(f"no images found in {args.data}")
        return 1

    # `-f path` needs a non-empty <stem>.gt.txt beside every image.
    bad = [
        p for p in images
        if not (p.with_suffix(".gt.txt").exists() and p.with_suffix(".gt.txt").read_text(encoding="utf-8").strip())
    ]
    if bad:
        print(f"{len(bad)} image(s) lack a non-empty .gt.txt (first: {bad[0]})")
        return 1

    shuffled = images[:]
    random.Random(args.seed).shuffle(shuffled)
    n_hold = max(1, round(len(shuffled) * args.holdout_fraction))
    holdout, train = shuffled[:n_hold], shuffled[n_hold:]

    args.train_out.write_text("\n".join(str(p) for p in sorted(train)) + "\n", encoding="utf-8")
    args.holdout_out.write_text("\n".join(str(p) for p in sorted(holdout)) + "\n", encoding="utf-8")
    print(f"train   : {len(train)} lines -> {args.train_out}")
    print(f"holdout : {len(holdout)} lines -> {args.holdout_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
