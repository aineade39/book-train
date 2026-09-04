#!/usr/bin/env python3
"""Modern, groupable/filterable HTML report over book-id-ios's per-spine
fuzzy-match telemetry (`<telemetry-dir>/runs/<runId>/run.json`).

Complements `report_book_id_telemetry.py` (which reports one row per *run*,
from `runs.ndjson`, for the "is the pipeline healthy" question) with one
row per *spine* (from `run.json`), for the "why did this particular fuzzy
match do that" question -- the level a human actually debugs at.

Field choices, per the fuzzy-match/OCR debugging conventions this repo's
own tools already encode (`overlay_spine_id.py`'s decision palette,
`report_ambiguous_spines.py`'s topCandidates + score/margin columns,
`§rerank-telemetry: "emit margin"` in `BookCatalogRoleAwareMatch.swift`):

  - No OBB geometry (cx/cy/w/h/angleDeg). This report is about *why the
    matcher decided what it decided*, not where the spine sits in the
    photo -- `overlay_spine_id.py` already covers geometry.
  - `candidates` (== `topCandidates`, winner included) is the single most
    useful "how many books did fuzzy match return" signal:
    `count == 1` is a clean, unambiguous win; `count > 1` means the
    catalog/query genuinely had several plausible titles in contention
    (worth checking whether the *right* one is even in the shortlist,
    which is a retrieval question, vs. ranked wrong, which is a scoring
    question); `count == 0` (no-match) means retrieval came back empty
    entirely -- a different bug class than either.
  - Score is never shown without margin next to it: an absolute score of
    e.g. 80 means "clear win" if the runner-up scored 40, and "coin flip"
    if the runner-up scored 79 -- margin is what actually separates
    "auto-accept" from "ambiguous" (`AcceptPolicy`'s 90/8 rule).
  - `assembledText` (raw OCR) stays next to the decision so a human can
    tell "OCR read garbage" (see `passedOCRQualityGate`) from "OCR was
    fine, the matcher picked wrong" at a glance -- two different bug
    classes that a bare score/decision column conflates.
  - Grouping (not just filtering) is what surfaces systemic issues, e.g.
    "most no-match rows share source=ocr and low ocrQualityScore" is
    obvious once grouped by decision, invisible in a flat sorted table.

No new dependency -- plain json/pathlib/f-strings + a self-contained
vanilla-JS table, matching every other tool in this directory.

Usage:
    .venv/bin/python3 tools/report_spine_matches.py --out spine_matches.html
    .venv/bin/python3 tools/report_spine_matches.py --dir ~/book-id-telemetry --limit 5 --out spine_matches.html
    .venv/bin/python3 tools/report_spine_matches.py --run 5CC9F9BC-... --out spine_matches.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Same default as report_book_id_telemetry.py -- BookID-macOS is
# deliberately unsandboxed (see book-id-ios/project.yml), so its telemetry
# lands at the plain, non-container Application Support path.
DEFAULT_TELEMETRY_DIR = Path.home() / "Library" / "Application Support" / "BookID" / "telemetry"

CAMERA_SCENE_ID = "camera"

# Matches overlay_spine_id.py's decision palette exactly, so a human
# moving between the geometry overlay and this table sees the same colors
# meaning the same thing.
DECISION_COLOR = {
    "auto-accept": "#2e7d32",
    "ambiguous": "#b8860b",
    "no-match": "#757575",
}


@dataclass
class SpineRow:
    scene: str
    run_id: str
    ts: str
    catalog: str
    source: str
    decision: str
    assembled_text: str
    ocr_quality: float | None
    passed_quality_gate: bool
    matched_title: str | None
    matched_author: str | None
    score: float | None
    margin: float | None
    candidates: list[str]
    detection_confidence: float | None

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def short_run(self) -> str:
        return self.run_id[:8] if self.run_id else "????????"

    @property
    def row_class(self) -> str:
        if self.decision == "no-match" and not self.passed_quality_gate:
            return "qgate"
        return {"auto-accept": "auto", "ambiguous": "ambig", "no-match": "nomatch"}.get(self.decision, "nomatch")

    @property
    def candidate_badge(self) -> str:
        n = self.candidate_count
        if n == 0:
            return "0 (empty retrieval)"
        if n == 1:
            return "1 (unique)"
        return f"{n} (contested)"


def resolve_dir(cli_dir: Path | None) -> Path:
    if cli_dir:
        return cli_dir.expanduser()
    env_dir = os.environ.get("BOOK_ID_TELEMETRY_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_TELEMETRY_DIR


def _ts_label(raw_ts: str | None) -> str:
    if not raw_ts:
        return "?"
    try:
        return datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).strftime("%m-%d %H:%M")
    except ValueError:
        return raw_ts


def load_run_dirs(telemetry_dir: Path, run_ids: list[str] | None, limit: int | None) -> list[Path]:
    runs_dir = telemetry_dir / "runs"
    if not runs_dir.exists():
        return []
    if run_ids:
        return [runs_dir / rid for rid in run_ids if (runs_dir / rid / "run.json").exists()]
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "run.json").exists()]
    dirs.sort(key=lambda d: (d / "run.json").stat().st_mtime, reverse=True)
    return dirs[:limit] if limit else dirs


def load_rows(telemetry_dir: Path, run_ids: list[str] | None, limit: int | None, scene_filter: str | None) -> list[SpineRow]:
    rows: list[SpineRow] = []
    for run_dir in load_run_dirs(telemetry_dir, run_ids, limit):
        try:
            payload: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
        except (json.JSONDecodeError, OSError):
            continue
        scene = payload.get("sceneId") or CAMERA_SCENE_ID
        if scene_filter and scene != scene_filter:
            continue
        for spine in payload.get("spines", []):
            rows.append(
                SpineRow(
                    scene=scene,
                    run_id=payload.get("runId", run_dir.name),
                    ts=_ts_label(payload.get("ts")),
                    catalog=payload.get("catalogSourceLabel", "?"),
                    source=spine.get("source", "?"),
                    decision=spine.get("decision", "no-match"),
                    assembled_text=spine.get("assembledText", ""),
                    ocr_quality=spine.get("ocrQualityScore"),
                    passed_quality_gate=bool(spine.get("passedOCRQualityGate", True)),
                    matched_title=spine.get("matchedTitle"),
                    matched_author=spine.get("matchedAuthor"),
                    score=spine.get("score"),
                    margin=spine.get("margin"),
                    candidates=spine.get("topCandidates") or [],
                    detection_confidence=spine.get("detectionConfidence"),
                )
            )
    return rows


# MARK: - HTML

COLUMNS: list[tuple[str, str, str]] = [
    # (key, header, kind) -- kind picks numeric vs text sort behavior.
    ("scene", "Scene", "text"),
    ("run", "Run", "text"),
    ("ts", "Time", "text"),
    ("decision", "Decision", "text"),
    ("candidateCount", "Candidates", "num"),
    ("source", "Source", "text"),
    ("assembledText", "OCR text", "text"),
    ("ocrQuality", "OCR quality", "num"),
    ("matchedTitle", "Matched title", "text"),
    ("matchedAuthor", "Matched author", "text"),
    ("score", "Score", "num"),
    ("margin", "Margin", "num"),
    ("detConf", "Det conf", "num"),
    ("candidates", "Candidate titles", "text"),
    ("catalog", "Catalog", "text"),
]


def _cell(row: SpineRow, key: str) -> str:
    if key == "scene":
        return row.scene
    if key == "run":
        return row.short_run
    if key == "ts":
        return row.ts
    if key == "decision":
        return row.decision
    if key == "candidateCount":
        return str(row.candidate_count)
    if key == "source":
        return row.source
    if key == "assembledText":
        return row.assembled_text
    if key == "ocrQuality":
        return "" if row.ocr_quality is None else f"{row.ocr_quality:.3f}"
    if key == "matchedTitle":
        return row.matched_title or ""
    if key == "matchedAuthor":
        return row.matched_author or ""
    if key == "score":
        return "" if row.score is None else f"{row.score:.1f}"
    if key == "margin":
        return "" if row.margin is None else f"{row.margin:.1f}"
    if key == "detConf":
        return "" if row.detection_confidence is None else f"{row.detection_confidence:.3f}"
    if key == "candidates":
        return " | ".join(row.candidates[:5])
    if key == "catalog":
        return row.catalog
    return ""


def render_summary_html(rows: list[SpineRow]) -> str:
    total = len(rows)
    if total == 0:
        return "<div class='meta'>No spines found.</div>"
    counts = {"auto-accept": 0, "ambiguous": 0, "no-match": 0}
    for r in rows:
        counts[r.decision if r.decision in counts else "no-match"] += 1
    cand_hist = {"0": 0, "1": 0, "2+": 0}
    for r in rows:
        n = r.candidate_count
        cand_hist["0" if n == 0 else ("1" if n == 1 else "2+")] += 1
    scenes = len({r.scene for r in rows})
    runs = len({r.run_id for r in rows})

    def pct(n: int) -> str:
        return f"{100.0 * n / total:.0f}%"

    parts = [
        f"<b>{total}</b> spines across <b>{runs}</b> run(s), <b>{scenes}</b> scene(s)",
        f"auto-accept <b>{counts['auto-accept']}</b> ({pct(counts['auto-accept'])})",
        f"ambiguous <b>{counts['ambiguous']}</b> ({pct(counts['ambiguous'])})",
        f"no-match <b>{counts['no-match']}</b> ({pct(counts['no-match'])})",
        f"candidates: 0&rarr;<b>{cand_hist['0']}</b>, 1&rarr;<b>{cand_hist['1']}</b>, 2+&rarr;<b>{cand_hist['2+']}</b>",
    ]
    return "<div class='meta'>" + " &nbsp;|&nbsp; ".join(parts) + "</div>"


def render_html(rows: list[SpineRow], out_path: Path, telemetry_dir: Path) -> None:
    thead = "".join(f'<th data-key="{k}" data-kind="{kind}">{label}</th>' for k, label, kind in COLUMNS)

    tbody_rows = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(_cell(row, k))}</td>" for k, _, _ in COLUMNS)
        data_attrs = (
            f'data-scene="{html.escape(row.scene)}" data-decision="{html.escape(row.decision)}" '
            f'data-source="{html.escape(row.source)}" data-run="{html.escape(row.run_id)}" '
            f'data-candbucket="{"0" if row.candidate_count == 0 else ("1" if row.candidate_count == 1 else "2+")}"'
        )
        tbody_rows.append(f'<tr class="{row.row_class}" {data_attrs}>{tds}</tr>')

    def facet_options(values: list[str]) -> str:
        uniq = sorted(set(values))
        return "".join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in uniq)

    scene_options = facet_options([r.scene for r in rows])
    run_options = facet_options([r.run_id for r in rows])

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spine match report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 16px 20px 40px; background: #fafafa; color: #1a1a1a; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .meta {{ color: #444; margin-bottom: 14px; font-size: 12.5px; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; padding: 10px 12px; background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; }}
  .toolbar label {{ display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }}
  .toolbar select, .toolbar input {{ font: 13px -apple-system, sans-serif; padding: 5px 8px; border: 1px solid #ccc; border-radius: 6px; background: #fff; }}
  .toolbar input#filter {{ width: 260px; }}
  .chipgroup {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .chip {{ padding: 4px 10px; border-radius: 999px; border: 1px solid #ccc; background: #fff; cursor: pointer; font-size: 12px; user-select: none; }}
  .chip.active {{ color: #fff; border-color: transparent; }}
  .chip[data-decision="auto-accept"].active {{ background: {DECISION_COLOR['auto-accept']}; }}
  .chip[data-decision="ambiguous"].active {{ background: {DECISION_COLOR['ambiguous']}; }}
  .chip[data-decision="no-match"].active {{ background: {DECISION_COLOR['no-match']}; }}
  .chip[data-decision="all"].active {{ background: #333; }}
  .count-badge {{ font-size: 11px; color: #777; margin-left: auto; white-space: nowrap; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.06); border-radius: 8px; overflow: hidden; }}
  th, td {{ border-bottom: 1px solid #ececec; padding: 6px 9px; vertical-align: top; text-align: left; white-space: nowrap; }}
  td {{ max-width: 320px; overflow: hidden; text-overflow: ellipsis; }}
  th {{ background: #f5f5f5; cursor: pointer; position: sticky; top: 0; font-weight: 600; z-index: 1; }}
  th:hover {{ background: #ededed; }}
  th.sorted::after {{ content: " " attr(data-dir); font-size: 10px; color: #888; }}
  tr.auto td {{ background: #f2faf2; }}
  tr.ambig td {{ background: #fffaf0; }}
  tr.nomatch td {{ background: #f7f7f7; }}
  tr.qgate td {{ background: #efefef; color: #888; }}
  tr.group-header td {{ background: #eef1f5; font-weight: 700; cursor: pointer; position: sticky; top: 29px; z-index: 1; border-top: 2px solid #dfe3e8; }}
  tr.group-collapsed {{ display: none; }}
  .decision-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>Spine match report ({len(rows)} rows)</h1>
{render_summary_html(rows)}
<div class="meta">Source: {html.escape(str(telemetry_dir))}. Colors: auto-accept green, ambiguous amber, no-match/quality-gate gray (same palette as overlay_spine_id.py). "Candidates" = fuzzy rerank's full shortlist, winner included -- 1 is a clean win, 2+ is contested, 0 is empty retrieval.</div>
<div class="toolbar">
  <label>Group by
    <select id="groupBy">
      <option value="none">None</option>
      <option value="scene">Scene</option>
      <option value="decision">Decision</option>
      <option value="source">Source</option>
      <option value="run">Run</option>
      <option value="candbucket">Candidate count</option>
    </select>
  </label>
  <label>Scene
    <select id="sceneFilter"><option value="">All</option>{scene_options}</select>
  </label>
  <label>Run
    <select id="runFilter"><option value="">All</option>{run_options}</select>
  </label>
  <label>Search
    <input id="filter" placeholder="OCR text, title, candidates…">
  </label>
  <div class="chipgroup" id="decisionChips">
    <span class="chip active" data-decision="all">All</span>
    <span class="chip" data-decision="auto-accept">Auto-accept</span>
    <span class="chip" data-decision="ambiguous">Ambiguous</span>
    <span class="chip" data-decision="no-match">No-match</span>
  </div>
  <span class="count-badge" id="countBadge"></span>
</div>
<table id="tbl">
<thead><tr>{thead}</tr></thead>
<tbody>
{''.join(tbody_rows)}
</tbody>
</table>
<script>
const tbl = document.getElementById('tbl');
const tbody = tbl.tBodies[0];
const allRows = [...tbody.rows];
const headers = [...tbl.querySelectorAll('th')];
const groupCols = {{scene: 'Scene', decision: 'Decision', source: 'Source', run: 'Run', candbucket: 'Candidate count'}};
let sortKey = null, sortAsc = true;
let activeDecision = 'all';

function currentFilters() {{
  return {{
    text: document.getElementById('filter').value.toLowerCase(),
    scene: document.getElementById('sceneFilter').value,
    run: document.getElementById('runFilter').value,
    decision: activeDecision,
  }};
}}

function rowMatches(row, f) {{
  if (f.scene && row.dataset.scene !== f.scene) return false;
  if (f.run && row.dataset.run !== f.run) return false;
  if (f.decision !== 'all' && row.dataset.decision !== f.decision) return false;
  if (f.text && !row.textContent.toLowerCase().includes(f.text)) return false;
  return true;
}}

function sortRows(rows) {{
  if (sortKey === null) return rows;
  const idx = headers.findIndex(h => h.dataset.key === sortKey);
  const kind = headers[idx].dataset.kind;
  const sorted = [...rows].sort((a, b) => {{
    const av = a.cells[idx].textContent.trim();
    const bv = b.cells[idx].textContent.trim();
    let cmp;
    if (kind === 'num') {{
      const an = parseFloat(av), bn = parseFloat(bv);
      cmp = (isNaN(an) ? -Infinity : an) - (isNaN(bn) ? -Infinity : bn);
    }} else {{
      cmp = av.localeCompare(bv);
    }}
    return sortAsc ? cmp : -cmp;
  }});
  return sorted;
}}

function render() {{
  const f = currentFilters();
  const visible = allRows.filter(r => rowMatches(r, f));
  const sorted = sortRows(visible);
  const groupBy = document.getElementById('groupBy').value;

  tbody.innerHTML = '';
  document.getElementById('countBadge').textContent = `${{visible.length}} / ${{allRows.length}} rows`;

  if (groupBy === 'none') {{
    sorted.forEach(r => tbody.appendChild(r));
    return;
  }}
  const groups = new Map();
  sorted.forEach(r => {{
    const key = r.dataset[groupBy] || '(none)';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }});
  const colCount = headers.length;
  [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0])).forEach(([key, groupRows]) => {{
    const header = document.createElement('tr');
    header.className = 'group-header';
    const td = document.createElement('td');
    td.colSpan = colCount;
    td.textContent = `${{groupCols[groupBy]}}: ${{key}}  (${{groupRows.length}})`;
    header.appendChild(td);
    header.addEventListener('click', () => {{
      groupRows.forEach(r => r.classList.toggle('group-collapsed'));
    }});
    tbody.appendChild(header);
    groupRows.forEach(r => tbody.appendChild(r));
  }});
}}

headers.forEach(th => {{
  th.addEventListener('click', () => {{
    if (sortKey === th.dataset.key) sortAsc = !sortAsc; else {{ sortKey = th.dataset.key; sortAsc = true; }}
    headers.forEach(h => {{ h.classList.remove('sorted'); h.removeAttribute('data-dir'); }});
    th.classList.add('sorted');
    th.setAttribute('data-dir', sortAsc ? '\\u25b2' : '\\u25bc');
    render();
  }});
}});

document.getElementById('groupBy').addEventListener('change', render);
document.getElementById('sceneFilter').addEventListener('change', render);
document.getElementById('runFilter').addEventListener('change', render);
document.getElementById('filter').addEventListener('input', render);
document.getElementById('decisionChips').addEventListener('click', (e) => {{
  const chip = e.target.closest('.chip');
  if (!chip) return;
  activeDecision = chip.dataset.decision;
  [...document.getElementById('decisionChips').children].forEach(c => c.classList.toggle('active', c === chip));
  render();
}});

render();
</script>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    print(f"Wrote {out_path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None, help="Telemetry dir (default: $BOOK_ID_TELEMETRY_DIR, else the macOS app's Application Support telemetry dir)")
    ap.add_argument("--run", action="append", default=None, help="Only this run id (repeatable). Default: all runs found, newest first.")
    ap.add_argument("--limit", type=int, default=None, help="Only the N most recently written runs (ignored with --run)")
    ap.add_argument("--scene", default=None, help="Only spines from this sceneId")
    ap.add_argument("--out", type=Path, required=True, help="Output HTML path")
    args = ap.parse_args()

    telemetry_dir = resolve_dir(args.dir)
    rows = load_rows(telemetry_dir, args.run, args.limit, args.scene)
    if not rows:
        print(f"No spines found under {telemetry_dir}/runs/**/run.json", file=sys.stderr)
    render_html(rows, args.out, telemetry_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
