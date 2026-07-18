#!/usr/bin/env python3
"""
Train YOLO-OBB on the combined 4TU + IEEE spine dataset (see
tools/build_spines_dataset.py) and export the result to Core ML.

Augmentation: degrees=90 + flipud=0.5 + fliplr=0.5 give the model rotated
spines "for free" every epoch (validated to fix 90-degree horizontal stacks
in an earlier smoke test); the merged dataset's rotated val subset then
scores checkpoints on angle robustness instead of only vertical accuracy.

Smoke test (fast, checks the merged labels/pipeline before the full run):
  python tools/train_combined_obb.py --smoke

Full run:
  python tools/train_combined_obb.py

Export only, from existing weights:
  python tools/train_combined_obb.py --skip-train --weights <path/to/best.pt>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derived_meta import dataset_tag_from_dir  # noqa: E402
from paths import derived_dir, models_candidates, runs_dir  # noqa: E402


# Naming convention (applies to both the run dir under --runs-out and the
# exported .mlpackage name), so a filename alone tells you what it is
# without opening args.yaml or results.csv:
#
#   {dataset}_{arch}_{imgsz}px_deg{degrees}_ep{epochs}_frac{pct}_{YYYYMMDD-HHMM}[_{tag}]
#
# e.g. 4tu-ieee_yolo26s-obb_1024px_deg90_ep20_frac15_20260716-2054_smoke
#
# The timestamp guarantees auto-generated names never collide across runs
# with different configs, so `exist_ok=True` can never silently overwrite a
# differently-configured run's weights/args.yaml (see MODELS.md for how that
# bit us with the old hardcoded "spine-obb" name).


def build_run_tag(data_yaml: Path, model: str, imgsz: int, degrees: float, epochs: int, fraction: float) -> str:
    dataset_tag = dataset_tag_from_dir(data_yaml)
    arch_tag = model.removesuffix(".pt")
    deg_tag = f"deg{int(degrees)}"
    frac_tag = f"frac{int(round(fraction * 100))}"
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{dataset_tag}_{arch_tag}_{imgsz}px_{deg_tag}_ep{epochs}_{frac_tag}_{ts}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=derived_dir("4tu-ieee_yolo-obb", "spines.yaml"),
    )
    p.add_argument("--model", default="yolo26s-obb.pt", help="Base checkpoint.")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=120)
    # 30 is too aggressive for a 120-epoch run with close_mosaic=10: the smoke
    # run's biggest single mAP jump happened right at the mosaic-close
    # transition (epoch 11/20). A plateau anywhere in the long mosaic-on phase
    # could early-stop the full run before it ever reaches that polish phase.
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--cache", default="disk", help="False, 'ram', or 'disk'.")
    p.add_argument("--degrees", type=float, default=90.0)
    p.add_argument("--flipud", type=float, default=0.5)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of train data to use.")
    p.add_argument("--runs-out", type=Path, default=runs_dir())
    p.add_argument("--name", default=None, help="Run name. Default: auto-generated from config + timestamp.")
    p.add_argument(
        "--export-out",
        type=Path,
        default=models_candidates(),
        help="Folder for exported .mlpackage (default: $BOOK_SPINES_DATA/models/candidates).",
    )
    p.add_argument("--export-name", default=None, help="Export filename. Default: <run-name>.mlpackage.")
    p.add_argument("--quantize", default="16", help="Core ML quantize: 16 (fp16), 8, or 'none'.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for a fast sanity run: epochs=20, fraction=0.15, patience=20.",
    )
    p.add_argument("--tag", default=None, help="Optional suffix appended to the auto-generated name, e.g. 'smoke'.")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--weights", type=Path, default=None, help="Existing .pt to export (implies --skip-train).")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 20
        args.fraction = 0.15
        args.patience = 20
        if args.tag is None:
            args.tag = "smoke"

    if args.name is None:
        args.name = build_run_tag(args.data_yaml, args.model, args.imgsz, args.degrees, args.epochs, args.fraction)
        if args.tag:
            args.name += f"_{args.tag}"

    if args.export_name is None:
        quant_tag = "fp16" if args.quantize == "16" else ("int8" if args.quantize == "8" else "raw")
        args.export_name = f"{args.name}_{quant_tag}.mlpackage"

    if args.weights is not None:
        args.skip_train = True
    return args


def git_commit_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def ensure_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in this Python.\n"
            "  source .venv-export/bin/activate\n"
            "  python -m pip install ultralytics opencv-python-headless"
        ) from exc
    return YOLO


def train(args: argparse.Namespace) -> Path:
    YOLO = ensure_ultralytics()
    args.runs_out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=args.cache,
        degrees=args.degrees,
        flipud=args.flipud,
        fliplr=args.fliplr,
        fraction=args.fraction,
        project=str(args.runs_out),
        name=args.name,
        exist_ok=True,
    )
    weights_dir = args.runs_out / args.name / "weights"
    best = weights_dir / "best.pt"
    if not best.is_file():
        best = weights_dir / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"No weights found under {weights_dir}")
    print(f"Best weights: {best}")
    return best


def export_coreml(
    weights: Path, export_out: Path, export_name: str, imgsz: int, quantize: str, args: argparse.Namespace
) -> Path:
    YOLO = ensure_ultralytics()
    export_out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    quant_arg: int | str | None = None if quantize.lower() == "none" else (int(quantize) if quantize.isdigit() else quantize)
    exported = model.export(format="coreml", imgsz=imgsz, nms=False, quantize=quant_arg)
    exported_path = Path(str(exported))
    dest = export_out / export_name
    if dest.exists():
        shutil.rmtree(dest)
    if exported_path.is_dir():
        shutil.copytree(exported_path, dest)
    else:
        shutil.copy2(exported_path, dest)
    print(f"Core ML package: {dest}")

    manifest = {
        "run_name": args.name,
        "export_name": export_name,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit_short(),
        "source_weights": str(weights),
        "run_dir": str(args.runs_out / args.name),
        "data_yaml": str(args.data_yaml),
        "model_base": args.model,
        "imgsz": imgsz,
        "epochs": args.epochs,
        "degrees": args.degrees,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "fraction": args.fraction,
        "quantize": quantize,
        "rotation_sweep": None,  # fill in via tools/eval_rotation_sweep.py --json and merge
    }
    manifest_path = export_out / (Path(export_name).stem + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    return dest


def main() -> int:
    args = parse_args()
    print(f"Run name: {args.name}")
    print(f"Export name: {args.export_name}")

    if not args.skip_train and not args.data_yaml.is_file():
        print(f"Missing {args.data_yaml}. Run tools/build_spines_dataset.py first.", file=sys.stderr)
        return 1

    weights = args.weights
    if not args.skip_train:
        weights = train(args)
    elif weights is None:
        guess = args.runs_out / args.name / "weights" / "best.pt"
        if guess.is_file():
            weights = guess
        else:
            print("No weights to export. Train first or pass --weights.", file=sys.stderr)
            return 1

    if not args.skip_export:
        assert weights is not None
        export_coreml(weights, args.export_out, args.export_name, args.imgsz, args.quantize, args)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
