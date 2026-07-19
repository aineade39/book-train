#!/usr/bin/env python3
"""Promote a Core ML export to the production default alias.

Default path (``--latest``):
  1. Find newest ``runs/*/weights/best.pt``
  2. Export to ``models/candidates/<run>_fp16.mlpackage`` if missing
  3. Rotation sweep vs ``SpineDetectorOBB-aug`` (reuse existing JSON if present)
  4. Delete any *real* package at the alias name (``SpineDetectorOBB.mlpackage``)
  5. Copy package + sidecar into ``models/production/`` and symlink the alias
  6. Update ``MODELS.md`` (Current best + Runs row with mAP50 from the sweep)

Sweep results are written to
``$BOOK_SPINES_DATA/eval/sweep-promote-<run>.json``. If that file (or another
``eval/sweep*.json`` containing the run label) already exists, promote reuses
it instead of re-running. Pass ``--force-sweep`` to recompute.

Leave ``SpineDetectorOBB-aug.mlpackage`` alone as the frozen acceptance baseline.

Usage:
  # Most recent run: export if needed, sweep (or reuse), then promote
  python tools/promote_coreml.py --latest

  # Explicit candidate
  python tools/promote_coreml.py \\
    --package $BOOK_SPINES_DATA/models/candidates/<run>_fp16.mlpackage

  python tools/promote_coreml.py --latest --skip-sweep   # install only
  python tools/promote_coreml.py --latest --force-sweep  # ignore cached JSON
  python tools/promote_coreml.py --latest --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    book_spines_data,
    eval_dir,
    models_candidates,
    models_production,
    runs_dir,
)

DEFAULT_ALIAS = "SpineDetectorOBB.mlpackage"
AUG_BASELINE = "SpineDetectorOBB-aug.mlpackage"
BUCKET_ORDER = ("original", "rot30", "rot45", "rot60", "rot90")
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_MD = REPO_ROOT / "MODELS.md"
QUANT_TAG = "fp16"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--latest",
        action="store_true",
        help="Promote newest runs/*/weights/best.pt (export first if no candidate Core ML).",
    )
    src.add_argument(
        "--package",
        type=Path,
        help="Candidate .mlpackage (path, or bare stem under models/candidates/).",
    )
    src.add_argument(
        "--run",
        type=str,
        help="Run directory name under runs/ (export if needed, then promote).",
    )
    p.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help=f"Stable production symlink name (default: {DEFAULT_ALIAS}).",
    )
    p.add_argument(
        "--map50",
        default=None,
        help='mAP50 string for MODELS.md (skips parsing sweep). e.g. "0.973 / 0.966 / …".',
    )
    p.add_argument(
        "--sweep-json",
        type=Path,
        default=None,
        help="Reuse an existing sweep JSON instead of running eval_rotation_sweep.py.",
    )
    p.add_argument(
        "--sweep-label",
        default=None,
        help="Top-level key in sweep JSON (default: run name / package stem without quant tag).",
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Do not run or require a rotation sweep (MODELS.md mAP50 stays TBD unless --map50).",
    )
    p.add_argument(
        "--force-sweep",
        action="store_true",
        help="Re-run eval_rotation_sweep.py even if a matching sweep JSON already exists.",
    )
    p.add_argument(
        "--device",
        default="mps",
        help="Ultralytics device for the sweep (default: mps).",
    )
    p.add_argument("--notes", default="", help="Notes cell for the MODELS.md row.")
    p.add_argument(
        "--no-models-md",
        action="store_true",
        help="Only install package + alias; do not edit MODELS.md.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite an existing production copy of this stem.")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="With --latest/--run: fail if candidate Core ML is missing instead of exporting.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without changing disk / MODELS.md.")
    return p.parse_args()


def latest_run_with_best() -> tuple[str, Path]:
    root = runs_dir()
    if not root.is_dir():
        raise SystemExit(f"No runs dir: {root}")
    best_hits: list[tuple[float, str, Path]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        best = run_dir / "weights" / "best.pt"
        if best.is_file():
            best_hits.append((best.stat().st_mtime, run_dir.name, best))
    if not best_hits:
        raise SystemExit(f"No runs/*/weights/best.pt under {root}")
    best_hits.sort(key=lambda t: t[0], reverse=True)
    _, name, path = best_hits[0]
    return name, path


def candidate_package_for_run(run_name: str) -> Path:
    return models_candidates(f"{run_name}_{QUANT_TAG}.mlpackage")


def prefer_python() -> Path:
    for rel in (".venv-export/bin/python", ".venv/bin/python"):
        cand = REPO_ROOT / rel
        if cand.is_file():
            return cand
    return Path(sys.executable)


def export_run(run_name: str, weights: Path, dry_run: bool) -> Path:
    dest = candidate_package_for_run(run_name)
    if dest.is_dir() and (dest.parent / f"{dest.stem}.json").is_file():
        print(f"Candidate already exported: {dest}", flush=True)
        return dest
    py = prefer_python()
    cmd = [
        str(py),
        str(REPO_ROOT / "tools" / "train_combined_obb.py"),
        "--skip-train",
        "--weights",
        str(weights),
        "--name",
        run_name,
        "--quantize",
        "16",
    ]
    print(f"Exporting Core ML for {run_name} …", flush=True)
    print("  " + " ".join(cmd), flush=True)
    if dry_run:
        return dest
    env = os.environ.copy()
    env.setdefault("BOOK_SPINES_DATA", str(book_spines_data()))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Export failed (exit {proc.returncode})")
    if not dest.is_dir():
        raise SystemExit(f"Export finished but missing {dest}")
    side = dest.parent / f"{dest.stem}.json"
    if not side.is_file():
        raise SystemExit(f"Export finished but missing sidecar {side}")
    return dest


def resolve_package(raw: Path) -> Path:
    raw = raw.expanduser()
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw.resolve())
    name = raw.name if raw.name.endswith(".mlpackage") else f"{raw.name}.mlpackage"
    candidates.append(models_candidates(name))
    if not raw.is_absolute():
        candidates.append((Path.cwd() / name).resolve())

    for path in candidates:
        if path.is_dir() and path.name.endswith(".mlpackage"):
            return path.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise SystemExit(f"Package not found. Tried: {tried}")


def load_sidecar(package: Path) -> dict[str, Any]:
    side = package.parent / f"{package.stem}.json"
    if not side.is_file():
        raise SystemExit(f"Missing sidecar (required): {side}")
    return json.loads(side.read_text())


def arch_from_sidecar(meta: dict[str, Any]) -> str:
    base = str(meta.get("model_base") or "unknown")
    return base.removesuffix(".pt")


def data_tag_from_sidecar(meta: dict[str, Any]) -> str:
    run = str(meta.get("run_name") or "")
    m = re.match(r"^([a-z0-9]+(?:-[a-z0-9]+)*)_yolo", run)
    if m:
        return m.group(1)
    yaml_path = Path(str(meta.get("data_yaml") or ""))
    parent = yaml_path.parent.name if yaml_path.name else ""
    parent = parent.removesuffix("_yolo-obb").removesuffix("-yolo-obb")
    return parent or "unknown"


def frac_cell(fraction: Any) -> str:
    try:
        f = float(fraction)
    except (TypeError, ValueError):
        return "?"
    return f"{int(round(f * 100))}%"


def deg_cell(degrees: Any) -> str:
    try:
        d = float(degrees)
    except (TypeError, ValueError):
        return "?"
    return str(int(d)) if d == int(d) else str(d)


def default_sweep_label(stem: str) -> str:
    for suffix in ("_fp16", "_int8", "_raw"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_sweep_path(label: str) -> Path:
    return eval_dir(f"sweep-promote-{label}.json")


def sweep_json_has_label(path: Path, label: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if label not in data or not isinstance(data[label], dict):
        return False
    return all(k in data[label] for k in BUCKET_ORDER)


def find_existing_sweep(label: str) -> Path | None:
    """Prefer the canonical promote path, then any eval/sweep*.json with the label."""
    primary = default_sweep_path(label)
    if sweep_json_has_label(primary, label):
        return primary
    eval_root = eval_dir()
    if not eval_root.is_dir():
        return None
    hits = sorted(
        (p for p in eval_root.glob("sweep*.json") if sweep_json_has_label(p, label)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def map50_from_sweep(sweep_path: Path, label: str) -> str:
    data = json.loads(sweep_path.read_text())
    if label not in data:
        keys = ", ".join(sorted(data.keys())) or "(none)"
        raise SystemExit(f"Label {label!r} not in {sweep_path}. Keys: {keys}")
    buckets = data[label]
    parts: list[str] = []
    for key in BUCKET_ORDER:
        if key not in buckets:
            raise SystemExit(f"Sweep label {label!r} missing bucket {key!r}")
        parts.append(f"{float(buckets[key]['map50']):.3f}")
    return " / ".join(parts)


def beats_baseline(sweep_path: Path, new_label: str, baseline_label: str) -> bool:
    data = json.loads(sweep_path.read_text())
    if new_label not in data or baseline_label not in data:
        return False
    for key in BUCKET_ORDER:
        if key not in data[new_label] or key not in data[baseline_label]:
            return False
        if float(data[new_label][key]["map50"]) < float(data[baseline_label][key]["map50"]):
            return False
    return True


def data_root_from_meta(meta: dict[str, Any]) -> Path:
    yaml_path = Path(str(meta.get("data_yaml") or "")).expanduser()
    if yaml_path.is_file():
        return yaml_path.parent
    fallback = book_spines_data() / "derived" / "4tu-ieee-shelves_yolo-obb"
    if (fallback / "spines.yaml").is_file():
        return fallback
    raise SystemExit(
        f"Cannot resolve dataset root from sidecar data_yaml={meta.get('data_yaml')!r}"
    )


def weights_from_meta(meta: dict[str, Any], run_name: str | None) -> Path:
    src = Path(str(meta.get("source_weights") or "")).expanduser()
    if src.is_file():
        return src
    if run_name:
        guess = runs_dir(run_name, "weights", "best.pt")
        if guess.is_file():
            return guess
    raise SystemExit(
        "Cannot find best.pt for sweep (sidecar source_weights missing and --run/--latest unknown)"
    )


def run_rotation_sweep(
    *,
    weights: Path,
    label: str,
    data_root: Path,
    imgsz: int,
    device: str,
    out_json: Path,
    dry_run: bool,
) -> Path:
    compare = models_production(AUG_BASELINE)
    if not compare.exists():
        raise SystemExit(f"Missing acceptance baseline: {compare}")
    py = prefer_python()
    cmd = [
        str(py),
        str(REPO_ROOT / "tools" / "eval_rotation_sweep.py"),
        "--weights",
        str(weights),
        "--weights-name",
        label,
        "--compare",
        str(compare),
        "--compare-name",
        "aug",
        "--data-root",
        str(data_root),
        "--angles",
        "30,45,60,90",
        "--imgsz",
        str(imgsz),
        "--device",
        device,
        "--json",
        str(out_json),
    ]
    print("Running rotation sweep …", flush=True)
    print("  " + " ".join(cmd), flush=True)
    if dry_run:
        return out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("BOOK_SPINES_DATA", str(book_spines_data()))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Rotation sweep failed (exit {proc.returncode})")
    if not out_json.is_file():
        raise SystemExit(f"Sweep finished but missing {out_json}")
    return out_json


def delete_real_alias(alias_path: Path, dry_run: bool) -> None:
    if alias_path.is_symlink():
        print(f"Removing existing alias symlink {alias_path.name}", flush=True)
        if not dry_run:
            alias_path.unlink()
        return
    if not alias_path.exists():
        return
    if not alias_path.is_dir():
        raise SystemExit(f"Alias path exists but is not a directory/symlink: {alias_path}")
    print(f"Deleting real package at alias name: {alias_path}", flush=True)
    if not dry_run:
        shutil.rmtree(alias_path)


def install_package(
    src: Path,
    dest: Path,
    sidecar_src: Path,
    alias_name: str,
    force: bool,
    dry_run: bool,
    sweep_meta: dict[str, Any] | None = None,
) -> Path:
    dest_side = dest.parent / f"{dest.stem}.json"
    if dest.exists() or dest.is_symlink():
        if not force:
            raise SystemExit(f"Already in production: {dest} (pass --force to replace)")
        print(f"Replacing existing {dest.name}", flush=True)
        if not dry_run:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
            if dest_side.is_file():
                dest_side.unlink()
    print(f"Copy {src.name} → {dest}", flush=True)
    if not dry_run:
        shutil.copytree(src, dest)
        shutil.copy2(sidecar_src, dest_side)
        meta = json.loads(dest_side.read_text())
        meta["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["production_alias"] = alias_name
        if sweep_meta is not None:
            meta["rotation_sweep"] = sweep_meta
        dest_side.write_text(json.dumps(meta, indent=2) + "\n")
        if sweep_meta is not None and sidecar_src.is_file():
            cand_meta = json.loads(sidecar_src.read_text())
            cand_meta["rotation_sweep"] = sweep_meta
            sidecar_src.write_text(json.dumps(cand_meta, indent=2) + "\n")
    return dest_side


def retarget_alias(alias_path: Path, target_name: str, dry_run: bool) -> None:
    print(f"Alias {alias_path.name} → {target_name}", flush=True)
    if dry_run:
        return
    if alias_path.is_symlink() or alias_path.is_file():
        alias_path.unlink()
    elif alias_path.exists():
        raise SystemExit(f"Refusing to replace non-symlink alias path: {alias_path}")
    alias_path.symlink_to(target_name)


def update_models_md(
    *,
    stem: str,
    meta: dict[str, Any],
    map50: str,
    notes: str,
    dry_run: bool,
) -> None:
    if not MODELS_MD.is_file():
        raise SystemExit(f"Missing {MODELS_MD}")
    text = MODELS_MD.read_text()

    arch = arch_from_sidecar(meta)
    data = data_tag_from_sidecar(meta)
    imgsz = meta.get("imgsz", "?")
    deg = deg_cell(meta.get("degrees"))
    ep = meta.get("epochs", "?")
    frac = frac_cell(meta.get("fraction"))
    note = notes.strip() or "Promoted; default alias `SpineDetectorOBB.mlpackage`."
    if "Current best" not in note and "**Current best.**" not in note:
        note = f"**Current best.** {note}"

    current_block = (
        f"## Current best\n\n"
        f"**`{stem}`** ({arch}, {data}, deg{deg}, {ep} ep) — production default via "
        f"`{DEFAULT_ALIAS}` symlink. Acceptance baseline remains "
        f"`SpineDetectorOBB-aug`.\n"
    )
    new_text, n = re.subn(
        r"## Current best\n\n.*?(?=\n## )",
        current_block,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise SystemExit("Could not locate ## Current best section in MODELS.md")

    row = (
        f"| `{stem}.mlpackage` | {arch} | {data} | {imgsz} | {deg} | {ep} | {frac} | "
        f"{map50} | {note} |"
    )
    table_header = (
        "| Artifact / run | Arch | Data | imgsz | deg | ep | frac | mAP50 | Notes |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    if table_header not in new_text:
        raise SystemExit("Could not locate Runs table header in MODELS.md")
    if f"`{stem}.mlpackage`" in new_text:
        new_text2, n_row = re.subn(
            rf"\| `{re.escape(stem)}\.mlpackage` \|.*\|\n",
            row + "\n",
            new_text,
            count=1,
        )
        if n_row != 1:
            raise SystemExit(f"Found stem mention but could not replace row for {stem}")
        new_text = new_text2
        print(f"MODELS.md: replaced row for {stem}.mlpackage", flush=True)
    else:
        new_text = new_text.replace(table_header, table_header + row + "\n", 1)
        print(f"MODELS.md: added row for {stem}.mlpackage", flush=True)

    def scrub_best(match: re.Match[str]) -> str:
        line = match.group(0)
        if f"`{stem}.mlpackage`" in line:
            return line
        return line.replace("**Current best.** ", "").replace("**Current best.**", "")

    new_text = re.sub(r"^\| `[^`]+` \|.*\|\s*$", scrub_best, new_text, flags=re.MULTILINE)

    print("MODELS.md: updated Current best", flush=True)
    if dry_run:
        print("--- MODELS.md preview (first Current best + new/updated row) ---")
        print(current_block)
        print(row)
        return
    MODELS_MD.write_text(new_text)


def resolve_source(args: argparse.Namespace) -> tuple[Path, str | None, Path | None]:
    """Return (package, run_name|None, weights|None)."""
    if args.package is not None:
        pkg = resolve_package(args.package)
        return pkg, None, None

    if args.latest:
        run_name, weights = latest_run_with_best()
        print(f"Latest run: {run_name}", flush=True)
        print(f"  weights: {weights}", flush=True)
    else:
        run_name = args.run
        assert run_name is not None
        weights = runs_dir(run_name, "weights", "best.pt")
        if not weights.is_file():
            raise SystemExit(f"Missing weights: {weights}")
        print(f"Run: {run_name}", flush=True)
        print(f"  weights: {weights}", flush=True)

    dest = candidate_package_for_run(run_name)
    if dest.is_dir() and (dest.parent / f"{dest.stem}.json").is_file():
        return dest.resolve(), run_name, weights
    if args.skip_export:
        raise SystemExit(f"No candidate export at {dest} (omit --skip-export to create it)")
    pkg = export_run(run_name, weights, args.dry_run)
    return pkg, run_name, weights


def main() -> int:
    args = parse_args()

    if args.dry_run and (args.latest or args.run) and args.package is None:
        if args.latest:
            run_name, weights = latest_run_with_best()
        else:
            run_name = args.run
            assert run_name is not None
            weights = runs_dir(run_name, "weights", "best.pt")
        dest = candidate_package_for_run(run_name)
        print(f"Latest/run: {run_name}", flush=True)
        print(f"  weights: {weights}", flush=True)
        if dest.is_dir():
            print(f"  candidate exists: {dest}", flush=True)
        else:
            print(f"  would export → {dest}", flush=True)
        if not args.skip_sweep and args.map50 is None:
            existing = None if args.force_sweep else find_existing_sweep(run_name)
            if existing is not None:
                print(f"  would reuse sweep JSON: {existing}", flush=True)
            else:
                print(
                    f"  would run rotation sweep vs {AUG_BASELINE} → {default_sweep_path(run_name)}",
                    flush=True,
                )
        alias_path = models_production(args.alias)
        if alias_path.exists() and not alias_path.is_symlink():
            print(f"  would delete real alias package: {alias_path}", flush=True)
        print("(dry-run)", flush=True)
        return 0

    src, run_name, weights = resolve_source(args)
    if args.dry_run and not src.exists():
        return 0

    stem = src.stem
    sidecar_src = src.parent / f"{stem}.json"
    meta = load_sidecar(src)
    if run_name is None:
        run_name = str(meta.get("run_name") or default_sweep_label(stem))
    if weights is None:
        try:
            weights = weights_from_meta(meta, run_name)
        except SystemExit:
            weights = None

    sweep_label = args.sweep_label or run_name or default_sweep_label(stem)
    sweep_path = args.sweep_json.expanduser() if args.sweep_json else None
    sweep_meta: dict[str, Any] | None = None
    map50 = args.map50
    reused_sweep = False

    if map50 is None and not args.skip_sweep:
        if sweep_path is None and not args.force_sweep:
            existing = find_existing_sweep(sweep_label)
            if existing is not None:
                sweep_path = existing
                reused_sweep = True
                print(f"Reusing existing sweep JSON: {sweep_path}", flush=True)

        if sweep_path is None or (args.force_sweep and args.sweep_json is None):
            if weights is None or not weights.is_file():
                raise SystemExit(
                    "Need best.pt for the default rotation sweep "
                    "(pass --sweep-json / --map50 / --skip-sweep)"
                )
            data_root = data_root_from_meta(meta)
            imgsz = int(meta.get("imgsz") or 1024)
            sweep_path = default_sweep_path(sweep_label)
            reused_sweep = False
            run_rotation_sweep(
                weights=weights,
                label=sweep_label,
                data_root=data_root,
                imgsz=imgsz,
                device=args.device,
                out_json=sweep_path,
                dry_run=args.dry_run,
            )
        elif args.sweep_json is not None:
            reused_sweep = True

        if not args.dry_run:
            assert sweep_path is not None
            map50 = map50_from_sweep(sweep_path, sweep_label)
            data = json.loads(sweep_path.read_text())
            won = beats_baseline(sweep_path, sweep_label, "aug")
            if not won and "aug" not in data:
                print(
                    f"Note: sweep JSON has no 'aug' compare key; "
                    f"cannot verify beat-{AUG_BASELINE} from this file alone.",
                    flush=True,
                )
            sweep_meta = {
                "path": str(sweep_path),
                "label": sweep_label,
                "compare": "aug",
                "map50": map50,
                "beats_aug_all_buckets": won,
                "reused": reused_sweep,
            }
            print(f"Sweep mAP50 ({sweep_label}): {map50}", flush=True)
            print(
                f"Beats {AUG_BASELINE} on every bucket: {'yes' if won else 'NO'}",
                flush=True,
            )
        else:
            map50 = "TBD"

    if map50 is None:
        map50 = "TBD"
        print("Warning: MODELS.md mAP50 will be TBD", file=sys.stderr)

    prod_dir = models_production()
    dest = prod_dir / f"{stem}.mlpackage"
    alias_path = prod_dir / args.alias

    print(f"Data root: {book_spines_data()}", flush=True)
    print(f"Source:    {src}", flush=True)
    print(f"Dest:      {dest}", flush=True)
    print(f"Alias:     {alias_path} → {stem}.mlpackage", flush=True)
    if args.dry_run:
        print("(dry-run)", flush=True)

    if not args.dry_run:
        prod_dir.mkdir(parents=True, exist_ok=True)

    delete_real_alias(alias_path, args.dry_run)
    install_package(
        src, dest, sidecar_src, args.alias, args.force, args.dry_run, sweep_meta=sweep_meta
    )
    retarget_alias(alias_path, f"{stem}.mlpackage", args.dry_run)

    if not args.no_models_md:
        if args.notes:
            notes = args.notes
        elif sweep_meta and sweep_meta.get("beats_aug_all_buckets"):
            notes = "beats aug on every bucket; production default alias"
        elif sweep_meta:
            notes = "production default alias; does NOT beat aug on every bucket"
        else:
            notes = "production default alias; rotation-sweep acceptance TBD"
        update_models_md(stem=stem, meta=meta, map50=map50, notes=notes, dry_run=args.dry_run)

    print("Done." if not args.dry_run else "Dry-run done.", flush=True)
    print(f"bookspines.swift default should resolve via: {alias_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
