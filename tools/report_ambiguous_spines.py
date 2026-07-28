#!/usr/bin/env python3
"""Build a sortable HTML table of ambiguous spine-id matches vs oracles."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ocr_parity import OBB, point_in_polygon  # noqa: E402
from spine_matching_parity_fixture import normalize_for_search  # noqa: E402

ORACLES_DIR = Path("/Users/joebr/dev/optimize-gemini/fixtures/oracles")
SCENES = ["bedroom1", "bookcase", "office1", "office2", "office3"]
HIT_THRESHOLD = 70.0


def fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(normalize_for_search(a), normalize_for_search(b))


def spine_obb(spine: dict) -> OBB:
    return OBB(
        float(spine["cx"]),
        float(spine["cy"]),
        float(spine["w"]),
        float(spine["h"]),
        math.radians(float(spine["angleDeg"])),
    )


def pair_oracle_indices(oracle: dict, spines: list[dict]) -> dict[int, int]:
    """Map spine index -> oracle book index by geometry (same as overlay_ocr_parity)."""
    books = oracle.get("books", [])
    if not books or not spines:
        return {}
    ow, oh = oracle["image"]["w"], oracle["image"]["h"]
    obbs = [spine_obb(s) for s in spines]
    avg_diag = sum(o.diag for o in obbs) / len(obbs) if obbs else 0.0
    radius = avg_diag if avg_diag > 0 else 200.0
    candidates: list[tuple[bool, float, int, int]] = []
    for bi, book in enumerate(books):
        y1000, x1000 = book["point_2d"]
        px, py = (x1000 / 1000.0) * ow, (y1000 / 1000.0) * oh
        for si, obb in enumerate(obbs):
            contained = point_in_polygon(px, py, obb.corners())
            dist = math.hypot(px - obb.cx, py - obb.cy)
            if contained or dist <= radius:
                candidates.append((not contained, dist, bi, si))
    candidates.sort()
    used_b: set[int] = set()
    used_s: set[int] = set()
    assign: dict[int, int] = {}
    for _, _, bi, si in candidates:
        if bi in used_b or si in used_s:
            continue
        used_b.add(bi)
        used_s.add(si)
        assign[si] = bi
    return assign


def load_rows(json_dir: Path, overrides: dict[str, Path]) -> list[dict]:
    rows: list[dict] = []
    for scene in SCENES:
        json_path = overrides.get(scene, json_dir / f"{scene}.json")
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text())
        oracle_path = ORACLES_DIR / f"{scene}.json"
        oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else None
        spines = payload.get("spines", [])
        oracle_by_spine: dict[int, int] = pair_oracle_indices(oracle, spines) if oracle else {}

        for si, spine in enumerate(spines):
            if spine.get("decision") != "ambiguous":
                continue
            tops = spine.get("topCandidates") or []
            top = tops[0] if tops else ""
            oracle_title = ""
            oracle_author = ""
            if oracle and si in oracle_by_spine:
                book = oracle["books"][oracle_by_spine[si]]
                oracle_title = book.get("title", "")
                oracle_author = book.get("author", "")
            oracle_fuzzy = fuzzy(top, oracle_title) if oracle_title else 0.0
            ocr_oracle_fuzzy = fuzzy(spine.get("assembledText", ""), oracle_title) if oracle_title else 0.0
            rows.append(
                {
                    "scene": scene,
                    "id": spine.get("id", "")[:8],
                    "ocr": spine.get("assembledText", ""),
                    "topCandidate": top,
                    "candidates": " | ".join(tops[:5]),
                    "matchScore": spine.get("score"),
                    "margin": spine.get("margin"),
                    "ocrQuality": spine.get("ocrQualityScore"),
                    "detConfidence": spine.get("detectionConfidence"),
                    "source": spine.get("source", ""),
                    "oracleTitle": oracle_title,
                    "oracleAuthor": oracle_author,
                    "oracleFuzzy": round(oracle_fuzzy, 1),
                    "ocrOracleFuzzy": round(ocr_oracle_fuzzy, 1),
                    "oracleHit": oracle_fuzzy >= HIT_THRESHOLD,
                    "hasOracle": bool(oracle_title),
                }
            )
    return rows


def render_html(rows: list[dict], out_path: Path) -> None:
    columns = [
        ("scene", "Scene"),
        ("ocr", "OCR text"),
        ("topCandidate", "Top candidate"),
        ("oracleTitle", "Oracle title"),
        ("oracleAuthor", "Oracle author"),
        ("matchScore", "Match score"),
        ("margin", "Margin"),
        ("ocrQuality", "OCR quality"),
        ("detConfidence", "Det conf"),
        ("oracleFuzzy", "Top↔oracle fuzzy"),
        ("ocrOracleFuzzy", "OCR↔oracle fuzzy"),
        ("oracleHit", "Oracle hit"),
        ("source", "Source"),
        ("candidates", "Top-5 candidates"),
        ("id", "Spine id"),
    ]

    def cell(key: str, row: dict) -> str:
        val = row.get(key, "")
        if isinstance(val, bool):
            return "yes" if val else "no"
        if isinstance(val, float):
            return f"{val:.3f}" if key in {"ocrQuality", "detConfidence"} else f"{val:.1f}"
        if val is None:
            return ""
        return html.escape(str(val))

    thead = "".join(f'<th data-key="{k}">{label}</th>' for k, label in columns)
    tbody_rows = []
    for row in rows:
        tds = "".join(f"<td>{cell(k, row)}</td>" for k, _ in columns)
        hit_cls = "hit" if row.get("oracleHit") else ("weak" if row.get("hasOracle") else "unpaired")
        tbody_rows.append(f'<tr class="{hit_cls}">{tds}</tr>')

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ambiguous spine matches</title>
<style>
  body {{ font: 13px/1.4 -apple-system, BlinkMacSystemFont, sans-serif; margin: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 8px; }}
  .meta {{ color: #555; margin-bottom: 12px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; text-align: left; }}
  th {{ background: #f4f4f4; cursor: pointer; position: sticky; top: 0; }}
  tr.hit td {{ background: #eef9ee; }}
  tr.weak td {{ background: #fff8e6; }}
  tr.unpaired td {{ background: #f7f7f7; }}
  td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(14) {{ max-width: 280px; word-break: break-word; }}
  .filters {{ margin: 8px 0 12px; }}
  input {{ padding: 4px 8px; width: 320px; }}
</style>
</head>
<body>
<h1>Ambiguous spine matches ({len(rows)} rows)</h1>
<div class="meta">Green = oracle fuzzy ≥ {HIT_THRESHOLD}. Yellow = oracle paired but below threshold. Gray = no oracle pair.</div>
<div class="filters"><label>Filter: <input id="filter" placeholder="search any column…"></label></div>
<table id="tbl">
<thead><tr>{thead}</tr></thead>
<tbody>
{''.join(tbody_rows)}
</tbody>
</table>
<script>
const tbl = document.getElementById('tbl');
const filter = document.getElementById('filter');
let sortCol = -1, sortAsc = true;
tbl.querySelectorAll('th').forEach((th, i) => {{
  th.addEventListener('click', () => {{
    if (sortCol === i) sortAsc = !sortAsc; else {{ sortCol = i; sortAsc = true; }}
    const rows = [...tbl.tBodies[0].rows];
    rows.sort((a, b) => {{
      const av = a.cells[i].textContent.trim();
      const bv = b.cells[i].textContent.trim();
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = (!Number.isNaN(an) && !Number.isNaN(bn)) ? an - bn : av.localeCompare(bv);
      return sortAsc ? cmp : -cmp;
    }});
    rows.forEach(r => tbl.tBodies[0].appendChild(r));
  }});
}});
filter.addEventListener('input', () => {{
  const q = filter.value.toLowerCase();
  [...tbl.tBodies[0].rows].forEach(r => {{
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}});
</script>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    print(f"Wrote {out_path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[], help="scene=path.json")
    args = ap.parse_args()
    overrides: dict[str, Path] = {}
    for item in args.override:
        scene, path = item.split("=", 1)
        overrides[scene] = Path(path)
    rows = load_rows(args.json_dir, overrides)
    render_html(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
