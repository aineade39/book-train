#!/usr/bin/env python3
"""Compact, grouped Markdown report over book-id-ios's file telemetry.

See book-id-ios's "Book ID Telemetry" plan (§Report design) and
`../book-id-ios/docs/TELEMETRY.md` for the artifact schemas this reads:

    <telemetry-dir>/
      runs.ndjson          # one JSON object per line, one per run
      runs/<runId>/run.json  # per-run phase timing + per-spine detail

Design, borrowed from this repo's own debug-report tools (see the plan's
"Conventions borrowed from this codebase's own tools" table):
  - A dedicated "Needs attention" section (report_ambiguous_spines.py) --
    only surface rows that need a human look, not just averages.
  - The auto/ambiguous/no-match/quality-gate funnel legend
    (overlay_spine_id.py) as the standard decision-count shape.
  - Fixed-width one-row-per-scene tables, decision buckets as columns
    (compare_ocr_parity.py's print_summary).
  - Summary bullets -> tables Markdown skeleton, sorted keys, no prose
    walls (run_shelf_detection.py's analysis.py).

No new dependency -- plain json/pathlib/f-strings, matching every other
tool in this directory.

Usage:
    .venv/bin/python3 tools/report_book_id_telemetry.py
    .venv/bin/python3 tools/report_book_id_telemetry.py --dir ~/book-id-telemetry --out report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# BookID-macOS is deliberately unsandboxed (see book-id-ios/project.yml),
# so its default telemetry directory is the plain, non-container
# Application Support path -- no Containers/<bundle-id> redirection.
DEFAULT_TELEMETRY_DIR = Path.home() / "Library" / "Application Support" / "BookID" / "telemetry"

# Thresholds for "Needs attention" -- fixed, simple heuristics, not a
# scoring model (per the plan: "good enough for a debug tool, easy to
# tune later").
SLOW_RUN_TOP_N = 3
JIGSAW_NO_GAIN_NEW_DETECTIONS = 0
HIGH_NOMATCH_RATIO = 0.2

CAMERA_SCENE_ID = "camera"


@dataclass
class Run:
    raw: dict[str, Any]
    run_json: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def run_id(self) -> str:
        return self.raw.get("runId", "")

    @property
    def short_id(self) -> str:
        return self.run_id[:4] if self.run_id else "????"

    @property
    def ok(self) -> bool:
        return bool(self.raw.get("ok"))

    @property
    def scene_id(self) -> str:
        return self.raw.get("sceneId") or CAMERA_SCENE_ID

    @property
    def ts(self) -> datetime | None:
        raw_ts = self.raw.get("ts")
        if not raw_ts:
            return None
        try:
            return datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    @property
    def ts_label(self) -> str:
        ts = self.ts
        return ts.strftime("%m-%d %H:%M") if ts else "?"

    @property
    def elapsed_ms(self) -> float | None:
        return self.raw.get("elapsedMs")

    @property
    def spine_count(self) -> int:
        return self.raw.get("spineCount") or 0

    @property
    def no_match(self) -> int:
        return self.raw.get("noMatch") or 0

    @property
    def used_jigsaw(self) -> bool:
        return bool(self.raw.get("usedJigsaw"))

    @property
    def new_detection_count(self) -> int:
        return self.raw.get("newDetectionCount") or 0

    @property
    def short_circuited(self) -> bool:
        return bool(self.raw.get("shortCircuited"))

    @property
    def path_label(self) -> str:
        if not self.ok:
            return "—"
        if self.short_circuited:
            return "barcode"
        if self.used_jigsaw:
            return f"jigsaw(+{self.new_detection_count})"
        return "single-shot"

    @property
    def dets_label(self) -> str:
        if not self.ok:
            return "—"
        first = self.raw.get("firstPassCount")
        final = self.spine_count
        if first is None:
            return f"—→{final}"
        return f"{first}→{final}"

    def slow_stage(self) -> tuple[str, float] | None:
        """The phase (other than `total`) with the largest `elapsedMs`,
        looked up from `run.json` only when it's already been loaded --
        see `load_run_json_for_slow_stage`."""
        if not self.run_json:
            return None
        phases = self.run_json.get("phases") or {}
        best: tuple[str, float] | None = None
        for name, entry in phases.items():
            if name == "total":
                continue
            ms = (entry or {}).get("elapsedMs")
            if ms is None:
                continue
            if best is None or ms > best[1]:
                best = (name, ms)
        return best

    def slow_stage_label(self) -> str:
        stage = self.slow_stage()
        if not stage:
            return "—"
        name, ms = stage
        return f"{name} ({ms:.0f})"


def resolve_dir(cli_dir: Path | None) -> Path:
    if cli_dir:
        return cli_dir.expanduser()
    env_dir = os.environ.get("BOOK_ID_TELEMETRY_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_TELEMETRY_DIR


def load_runs(telemetry_dir: Path) -> list[Run]:
    ndjson_path = telemetry_dir / "runs.ndjson"
    if not ndjson_path.exists():
        return []
    runs: list[Run] = []
    for line in ndjson_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(Run(raw=json.loads(line)))
        except json.JSONDecodeError:
            continue
    return runs


def load_run_json_for_slow_stage(telemetry_dir: Path, runs: list[Run]) -> None:
    """Opens `run.json` only for successful runs -- cheap either way
    (files are tens of KB), but skipped for failures, which never get one."""
    for run in runs:
        if not run.ok or not run.run_id:
            continue
        run_json_path = telemetry_dir / "runs" / run.run_id / "run.json"
        if not run_json_path.exists():
            continue
        try:
            run.run_json = json.loads(run_json_path.read_text())
        except json.JSONDecodeError:
            continue


# MARK: - Summary

def render_summary(runs: list[Run]) -> list[str]:
    ok_runs = [r for r in runs if r.ok]
    failed_runs = [r for r in runs if not r.ok]
    lines = [f"- {len(ok_runs)} ok, {len(failed_runs)} failed"]

    catalogs = Counter(r.raw.get("catalogSourceLabel", "?") for r in runs)
    if catalogs:
        catalog_str = ", ".join(f"{label} ({count} runs)" for label, count in catalogs.most_common())
        lines.append(f"- Catalogs: {catalog_str}")

    paths = Counter(r.path_label for r in ok_runs)
    if paths:
        # Collapse every distinct "jigsaw(+N)" label into one "jigsaw" bucket
        # for the summary line -- the per-scene tables keep the detail.
        collapsed: Counter[str] = Counter()
        for label, count in paths.items():
            key = "jigsaw" if label.startswith("jigsaw") else label
            collapsed[key] += count
        path_str = ", ".join(f"{label} ({count})" for label, count in collapsed.most_common())
        lines.append(f"- Path mix: {path_str}")

    total_spines = sum(r.spine_count for r in ok_runs)
    auto = sum(r.raw.get("autoAccepted") or 0 for r in ok_runs)
    amb = sum(r.raw.get("ambiguous") or 0 for r in ok_runs)
    none_ = sum(r.raw.get("noMatch") or 0 for r in ok_runs)
    qgate = sum(r.raw.get("didNotPassQualityGate") or 0 for r in ok_runs)
    if total_spines:
        auto_pct = 100.0 * auto / total_spines
        lines.append(
            f"- Match funnel: {auto} auto / {amb} ambiguous / {none_} no-match / {qgate} quality-gate "
            f"({total_spines} spines, {auto_pct:.1f}% auto)"
        )

    stage_totals: Counter[str] = Counter()
    total_elapsed = 0.0
    for r in ok_runs:
        if not r.run_json:
            continue
        phases = r.run_json.get("phases") or {}
        total_ms = (phases.get("total") or {}).get("elapsedMs")
        if not total_ms:
            continue
        total_elapsed += total_ms
        for name, entry in phases.items():
            if name == "total":
                continue
            stage_totals[name] += (entry or {}).get("elapsedMs") or 0
    if stage_totals and total_elapsed > 0:
        dominant, dominant_ms = stage_totals.most_common(1)[0]
        pct = 100.0 * dominant_ms / total_elapsed
        lines.append(f"- Dominant slow stage across runs: {dominant} (avg {pct:.0f}% of total elapsedMs)")

    return lines


# MARK: - Needs attention

def render_needs_attention(runs: list[Run]) -> list[str]:
    ok_runs = [r for r in runs if r.ok]
    rows: list[str] = []

    for r in [r for r in runs if not r.ok]:
        error = r.raw.get("errorMessage") or "(no message)"
        rows.append(f"- FAILED  {r.short_id}  {r.ts_label}  {r.raw.get('catalogSourceLabel', '?'):<7}  \"{error}\"")

    slowest = sorted((r for r in ok_runs if r.elapsed_ms is not None), key=lambda r: r.elapsed_ms or 0, reverse=True)
    for r in slowest[:SLOW_RUN_TOP_N]:
        stage = r.slow_stage()
        stage_str = f" ({stage[0]} {stage[1]:.0f}ms)" if stage else ""
        rows.append(f"- SLOW    {r.short_id}  {r.ts_label}  {r.raw.get('catalogSourceLabel', '?'):<7}  {r.elapsed_ms:.0f}ms{stage_str}")

    for r in ok_runs:
        if r.used_jigsaw and r.new_detection_count == JIGSAW_NO_GAIN_NEW_DETECTIONS:
            rows.append(
                f"- JIGSAW-NO-GAIN  {r.short_id}  {r.ts_label}  {r.raw.get('catalogSourceLabel', '?'):<7}  "
                f"jigsaw ran, +0 new detections"
            )

    for r in ok_runs:
        if r.spine_count > 0 and (r.no_match / r.spine_count) > HIGH_NOMATCH_RATIO:
            pct = 100.0 * r.no_match / r.spine_count
            rows.append(
                f"- HIGH-NOMATCH     {r.short_id}  {r.ts_label}  {r.raw.get('catalogSourceLabel', '?'):<7}  "
                f"{r.no_match}/{r.spine_count} spines no-match ({pct:.0f}%)"
            )

    return rows if rows else ["- (none)"]


# MARK: - Scene tables

SCENE_TABLE_HEADER = "| run  | ts           | catalog | path        | dets(1st→final) | spines | auto | amb | none | qgate | total_ms | slow stage  |"
SCENE_TABLE_SEP = "|------|--------------|---------|-------------|------------------|--------|------|-----|------|-------|----------|-------------|"


def render_scene_table_row(r: Run) -> str:
    total_ms = f"{r.elapsed_ms:.0f}" if r.elapsed_ms is not None else "—"
    return (
        f"| {r.short_id} | {r.ts_label:<12} | {r.raw.get('catalogSourceLabel', '?'):<7} | {r.path_label:<11} "
        f"| {r.dets_label:<16} | {r.spine_count:<6} | {r.raw.get('autoAccepted') or 0:<4} "
        f"| {r.raw.get('ambiguous') or 0:<3} | {r.raw.get('noMatch') or 0:<4} | {r.raw.get('didNotPassQualityGate') or 0:<5} "
        f"| {total_ms:<8} | {r.slow_stage_label():<11} |"
    )


def _sort_ts(r: Run) -> datetime:
    return r.ts.replace(tzinfo=None) if r.ts else datetime.min


def render_scenes(runs: list[Run]) -> list[str]:
    grouped: dict[str, list[Run]] = {}
    for r in runs:
        grouped.setdefault(r.scene_id, []).append(r)

    named_scenes = sorted((s for s in grouped if s != CAMERA_SCENE_ID), key=lambda s: s.lower())
    distinct_count = len(named_scenes) + (1 if CAMERA_SCENE_ID in grouped else 0)

    lines = [f"## Scenes (grouped by sceneId, {distinct_count} distinct; \"{CAMERA_SCENE_ID}\" runs listed individually)", ""]

    for scene in named_scenes:
        scene_runs = sorted(grouped[scene], key=_sort_ts)
        lines.append(f"### {scene} ({len(scene_runs)} run{'s' if len(scene_runs) != 1 else ''})")
        lines.append(SCENE_TABLE_HEADER)
        lines.append(SCENE_TABLE_SEP)
        for r in scene_runs:
            lines.append(render_scene_table_row(r))
        lines.append("")

    if CAMERA_SCENE_ID in grouped:
        camera_runs = sorted(grouped[CAMERA_SCENE_ID], key=_sort_ts)
        lines.append(f"### {CAMERA_SCENE_ID} (1-off live captures, {len(camera_runs)} runs)")
        lines.append(SCENE_TABLE_HEADER)
        lines.append(SCENE_TABLE_SEP)
        for r in camera_runs:
            lines.append(render_scene_table_row(r))
        lines.append("")

    return lines


# MARK: - Failed runs

def render_failed_runs(runs: list[Run]) -> list[str]:
    failed = [r for r in runs if not r.ok]
    if not failed:
        return []
    lines = [
        "## Failed runs",
        "| run  | ts           | catalog | error                              |",
        "|------|--------------|---------|-------------------------------------|",
    ]
    for r in failed:
        error = (r.raw.get("errorMessage") or "").replace("\n", " ")
        lines.append(f"| {r.short_id} | {r.ts_label:<12} | {r.raw.get('catalogSourceLabel', '?'):<7} | {error:<35} |")
    lines.append("")
    return lines


# MARK: - Top level

def build_report(runs: list[Run], telemetry_dir: Path) -> str:
    dated = [r.ts for r in runs if r.ts is not None]
    date_range = ""
    if dated:
        lo, hi = min(dated).date(), max(dated).date()
        date_range = f" ({len(runs)} runs, {lo}..{hi})" if lo != hi else f" ({len(runs)} runs, {lo})"
    elif runs:
        date_range = f" ({len(runs)} runs)"

    lines = [
        "# Book ID Telemetry Report",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%dT%H:%M')} from {telemetry_dir / 'runs.ndjson'}{date_range}",
        "",
    ]

    if not runs:
        lines.append("No runs recorded yet.")
        return "\n".join(lines) + "\n"

    lines.append("## Summary")
    lines.extend(render_summary(runs))
    lines.append("")

    lines.append("## Needs attention")
    lines.extend(render_needs_attention(runs))
    lines.append("")

    lines.extend(render_scenes(runs))
    lines.extend(render_failed_runs(runs))

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dir", type=Path, default=None,
        help="Telemetry directory containing runs.ndjson + runs/ (default: $BOOK_ID_TELEMETRY_DIR, "
        "else the macOS app's Application Support telemetry directory)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write Markdown here instead of stdout")
    args = ap.parse_args()

    telemetry_dir = resolve_dir(args.dir)
    runs = load_runs(telemetry_dir)
    load_run_json_for_slow_stage(telemetry_dir, runs)
    report = build_report(runs, telemetry_dir)

    if args.out:
        args.out.write_text(report)
        print(f"Wrote {args.out} ({len(runs)} runs)", file=sys.stderr)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
