#!/usr/bin/env python3
"""
Build images/val_train + labels/val_train for every-epoch training validation.

Keeps full images/val (incl. _rot* acceptance buckets) untouched for
eval_rotation_sweep.py. Train-time val is a stratified unrotated subset
(~350–400 by default) so epoch wall-clock stays reasonable on GPU.

Usage:
  .venv/bin/python tools/make_val_train_subset.py
  .venv/bin/python tools/make_val_train_subset.py --count 120 --tag smoke
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import derived_dir  # noqa: E402


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def source_bucket(stem: str) -> str:
    s = stem.lower()
    if "_rot" in s:
        return "rotated"
    if s.startswith("ieee_"):
        return "ieee"
    if s.startswith("4tu_") or s.startswith("tu_") or s.startswith("spine_"):
        return "4tu"
    if s.startswith("rf_"):
        return "roboflow"
    if s.startswith("os_") or "open" in s or "shelves" in s:
        return "open-shelves"
    return "other"


def list_val_images(val_img: Path) -> list[Path]:
    return sorted(p for p in val_img.iterdir() if p.suffix.lower() in IMG_EXTS)


def stratified_pick(images: list[Path], count: int, seed: int) -> list[Path]:
    unrot = [p for p in images if source_bucket(p.stem) != "rotated"]
    by: dict[str, list[Path]] = defaultdict(list)
    for p in unrot:
        by[source_bucket(p.stem)].append(p)
    for v in by.values():
        v.sort(key=lambda p: p.name)

    if count >= len(unrot):
        return unrot

    rng = random.Random(seed)
    # Proportional allocation, then fill remainder from largest leftovers.
    quotas: dict[str, int] = {}
    assigned = 0
    for src, items in by.items():
        q = int(round(count * len(items) / len(unrot)))
        q = min(q, len(items))
        quotas[src] = q
        assigned += q
    # Fix rounding drift
    order = sorted(by.keys(), key=lambda s: len(by[s]), reverse=True)
    while assigned < count:
        for src in order:
            if quotas[src] < len(by[src]):
                quotas[src] += 1
                assigned += 1
                if assigned >= count:
                    break
    while assigned > count:
        for src in reversed(order):
            if quotas[src] > 0:
                quotas[src] -= 1
                assigned -= 1
                if assigned <= count:
                    break

    picked: list[Path] = []
    for src, items in by.items():
        pool = items[:]
        rng.shuffle(pool)
        picked.extend(pool[: quotas[src]])
    picked.sort(key=lambda p: p.name)
    return picked


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def write_yaml(out_root: Path, val_rel: str, yaml_name: str) -> Path:
    path = out_root / yaml_name
    path.write_text(
        "\n".join(
            [
                f"path: {out_root}",
                "train: images/train",
                f"val: {val_rel}",
                "names:",
                "  0: spine",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def build_val_train_subset(
    root: Path,
    *,
    count: int = 375,
    seed: int = 42,
    split_name: str = "val_train",
    yaml_name: str = "spines_train.yaml",
) -> int:
    val_img = root / "images" / "val"
    val_lab = root / "labels" / "val"
    if not val_img.is_dir():
        raise FileNotFoundError(f"Missing {val_img}")

    images = list_val_images(val_img)
    picked = stratified_pick(images, count, seed)
    out_img = root / "images" / split_name
    out_lab = root / "labels" / split_name
    if out_img.exists():
        shutil.rmtree(out_img)
    if out_lab.exists():
        shutil.rmtree(out_lab)
    out_img.mkdir(parents=True)
    out_lab.mkdir(parents=True)

    missing_labels = 0
    by_src: dict[str, int] = defaultdict(int)
    for img in picked:
        lab = val_lab / f"{img.stem}.txt"
        if not lab.is_file():
            missing_labels += 1
            continue
        link_or_copy(img, out_img / img.name)
        link_or_copy(lab, out_lab / lab.name)
        by_src[source_bucket(img.stem)] += 1

    n = sum(1 for _ in out_img.iterdir())
    yaml_path = write_yaml(root, f"images/{split_name}", yaml_name)
    write_yaml(root, "images/val", "spines.yaml")

    print(f"val_train: {n} images (requested {count}) -> {out_img}")
    print(f"  by source: {dict(sorted(by_src.items()))}")
    if missing_labels:
        print(f"  skipped (missing labels): {missing_labels}")
    print(f"train YAML: {yaml_path} (val={split_name})")
    print(f"full YAML:  {root / 'spines.yaml'} (val=images/val, untouched)")
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--derived",
        type=Path,
        default=derived_dir("4tu-ieee-shelves_yolo-obb"),
        help="Combined derived dataset root.",
    )
    p.add_argument("--count", type=int, default=375, help="Unrotated val_train size (default 375).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-name",
        default="val_train",
        help="images/<name> + labels/<name> for the subset (default val_train).",
    )
    p.add_argument(
        "--yaml-name",
        default="spines_train.yaml",
        help="YAML written for training (full val stays in spines.yaml).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    build_val_train_subset(
        args.derived,
        count=args.count,
        seed=args.seed,
        split_name=args.split_name,
        yaml_name=args.yaml_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
