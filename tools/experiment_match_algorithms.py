#!/usr/bin/env python3
"""Experiment: which fuzzy string-matching algorithm best recovers the
correct book title from raw OCR text, evaluated on real bookcase.jpg
telemetry against its oracle ground truth.

## Literature consulted (external experts)

No single algorithm is universally best (Waradana et al. 2025, "Comparative
Performance Analysis of String Matching Algorithms... on Academic Datasets";
Gunawansyah et al. 2023, IEEE TSSA) -- the right choice depends on string
length and the kind of noise. Consistent findings across multiple
practitioner/vendor writeups (Flagright's AML-screening comparison,
DataLadder's "Fuzzy Matching 101", MatchDataPro's fuzzy-matching guide) and
RapidFuzz's own docs/benchmarks:

  - Jaro-Winkler: best for short strings/proper names, rewards shared
    prefixes -- standard in census/KYC record linkage.
  - Levenshtein/`ratio`: the standard edit-distance baseline for
    typo-level noise in fixed-length strings.
  - Token Sort/Set Ratio: preferred once word order can differ or one
    string is a superset of the other's words -- exactly the OCR case
    of a multi-line, mashed spine-text blob vs. a clean catalog title.
  - WRatio: a length-adaptive weighted combination of the above,
    recommended as a strong default "when unsure."
  - BM25/FTS rank alone: the retrieval-only baseline -- useful to check
    whether reranking is earning its keep over plain lexical rank.

This repo already has its own internal research on this exact question:
`FieldAwareScore.swift`'s doc comment records a `A/B` against all 5 oracle
scenes for a "coverage-adjusted token_set_ratio" variant that *regressed*
top-candidate oracle-hit rate from 62.6% to 48.5% -- i.e. token_set_ratio
(the family recommended above for OCR-blob-vs-title matching) is already
the tuned production choice, not a naive pick. This script is an
independent, from-scratch check of that choice against the alternatives
above, isolated from `FieldAwareScore`'s field-splitting/popularity-prior
design (see "Methodology" below) so the *raw scoring metric* can be judged
on its own.

## Methodology

1. Geometrically pair `run.json`'s detected spines to `bookcase.json`'s
   oracle books (same nearest-OBB-center approach as
   `report_ambiguous_spines.py`/`overlay_ocr_parity.py`) -- gives each
   spine a ground-truth title/author independent of anything the matcher
   itself produced.
2. For each paired spine that passed the OCR quality gate, retrieve a
   *shared* candidate shortlist from the catalog via a permissive
   (OR-of-tokens) FTS5 MATCH, ranked by SQLite's built-in `bm25()` --
   identical shortlist for every algorithm under test, so only the
   *scoring* step differs between rows in the results table.
3. Score every shortlist candidate's `"title author"` blob against the
   raw OCR text with each candidate algorithm; take the argmax per
   algorithm.
4. Judge each algorithm's pick against the oracle title with two
   criteria that use neither RapidFuzz nor any algorithm under test
   (so the judge can't structurally favor one contender): normalized
   exact-match, and normalized substring-containment (looser, catches
   subtitle truncation like "Cascade Alpine Guide" vs "...: Columbia
   River to Stevens Pass").
5. `production` is a reference row: the real `matchedTitle`/
   `topCandidates[0]` `run.json` already recorded via the full
   `FieldAwareScore` (field-split + popularity prior) pipeline -- not
   apples-to-apples with the single-blob algorithms above, but shows
   where the shipped system already lands.

Usage:
    .venv/bin/python3 tools/experiment_match_algorithms.py
    .venv/bin/python3 tools/experiment_match_algorithms.py --run-json <path> --out report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rapidfuzz import distance, fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ocr_parity import OBB, point_in_polygon  # noqa: E402

DEFAULT_TELEMETRY_DIR = Path.home() / "Library" / "Application Support" / "BookID" / "telemetry"
DEFAULT_ORACLE = Path("/Users/joebr/dev/optimize-gemini/fixtures/oracles/bookcase.json")
DEFAULT_DB = Path.home() / "ml" / "book-spines" / "derived" / "book-catalog" / "ios_en.sqlite"
MIN_FTS_TOKEN_LEN = 3  # SQLite trigram tokenizer can't index shorter tokens.


def normalize_for_search(raw: str) -> str:
    """Same rules as `Sources/SpineMatching/Normalization.swift` (see
    `spine_matching_parity_fixture.normalize_for_search`, duplicated here
    to keep this experiment a single, standalone file)."""
    folded = unicodedata.normalize("NFKD", raw)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.casefold()
    out: list[str] = []
    last_was_space = False
    for ch in folded:
        if ch.isspace():
            if not last_was_space and out:
                out.append(" ")
            last_was_space = True
            continue
        last_was_space = False
        if ch.isalnum() or ch == " ":
            out.append(ch)
    return "".join(out).strip()


# MARK: - Oracle pairing (geometry-based, same approach as report_ambiguous_spines.py)

def spine_obb(spine: dict) -> OBB:
    return OBB(float(spine["cx"]), float(spine["cy"]), float(spine["w"]), float(spine["h"]), math.radians(float(spine["angleDeg"])))


def pair_oracle_indices(oracle: dict, spines: list[dict]) -> dict[int, int]:
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


# MARK: - Candidate retrieval (shared shortlist across every algorithm)

@dataclass
class Candidate:
    title: str
    author: str
    work_key: str
    popularity_rank: int | None

    @property
    def blob(self) -> str:
        return normalize_for_search(f"{self.title} {self.author}")


def fts_shortlist(conn: sqlite3.Connection, ocr_text: str, cap: int) -> list[Candidate]:
    tokens = [t for t in normalize_for_search(ocr_text).split() if len(t) >= MIN_FTS_TOKEN_LEN]
    if not tokens:
        return []
    # Deliberately permissive OR-of-tokens -- the point of this experiment
    # is to compare *scoring*, so retrieval should not itself gatekeep
    # which algorithms even get a chance at the right answer.
    query = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
    try:
        rows = conn.execute(
            """
            SELECT b.title, b.author, b.workKey, b.popularityRank
            FROM books_fts f
            JOIN books b ON b.id = f.rowid
            WHERE books_fts MATCH ?
            ORDER BY bm25(books_fts)
            LIMIT ?
            """,
            (query, cap),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [Candidate(title=r[0], author=r[1], work_key=r[2], popularity_rank=r[3]) for r in rows]


# MARK: - Algorithms under test

@dataclass
class Algorithm:
    name: str
    score: Callable[[str, str], float]  # (ocr_text, candidate_blob) -> 0..100, higher is better


def jaro_winkler_score(a: str, b: str) -> float:
    return distance.JaroWinkler.normalized_similarity(a, b) * 100


ALGORITHMS = [
    Algorithm("bm25_rank_only", lambda a, b: 0.0),  # special-cased: just takes shortlist[0]
    Algorithm("ratio (Levenshtein)", fuzz.ratio),
    Algorithm("jaro_winkler", jaro_winkler_score),
    Algorithm("token_sort_ratio", fuzz.token_sort_ratio),
    Algorithm("token_set_ratio", fuzz.token_set_ratio),
    Algorithm("partial_ratio", fuzz.partial_ratio),
    Algorithm("WRatio", fuzz.WRatio),
]


@dataclass
class SpineEval:
    spine_id: str
    ocr_text: str
    oracle_title: str
    oracle_author: str
    shortlist_size: int
    picks: dict[str, Candidate | None] = field(default_factory=dict)
    production_title: str | None = None


def is_hit(oracle_title: str, picked_title: str | None) -> tuple[bool, bool]:
    """(exact, contains) -- both judges are string-normalization only, no
    fuzzy library involved, so they can't structurally favor any
    algorithm under test."""
    if not picked_title:
        return False, False
    o = normalize_for_search(oracle_title)
    p = normalize_for_search(picked_title)
    if not o or not p:
        return False, False
    exact = o == p
    contains = (o in p or p in o) and min(len(o), len(p)) >= 4
    return exact, contains


def load_spines(run_json_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(run_json_path.read_text())
    return payload, payload.get("spines", [])


def run_experiment(run_json_path: Path, oracle_path: Path, db_path: Path, shortlist_cap: int) -> list[SpineEval]:
    payload, spines = load_spines(run_json_path)
    oracle = json.loads(oracle_path.read_text())
    pairing = pair_oracle_indices(oracle, spines)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    evals: list[SpineEval] = []
    try:
        for si, spine in enumerate(spines):
            if si not in pairing:
                continue
            if not spine.get("passedOCRQualityGate", True):
                continue
            ocr_text = spine.get("assembledText", "")
            if not ocr_text.strip():
                continue
            book = oracle["books"][pairing[si]]
            ev = SpineEval(
                spine_id=spine.get("id", "")[:8],
                ocr_text=ocr_text,
                oracle_title=book.get("title", ""),
                oracle_author=book.get("author", ""),
                shortlist_size=0,
                production_title=spine.get("matchedTitle") or (spine.get("topCandidates") or [None])[0],
            )
            shortlist = fts_shortlist(conn, ocr_text, shortlist_cap)
            ev.shortlist_size = len(shortlist)
            ocr_normalized = normalize_for_search(ocr_text)
            for algo in ALGORITHMS:
                if not shortlist:
                    ev.picks[algo.name] = None
                    continue
                if algo.name == "bm25_rank_only":
                    ev.picks[algo.name] = shortlist[0]
                    continue
                best = max(shortlist, key=lambda c: algo.score(ocr_normalized, c.blob))
                ev.picks[algo.name] = best
            evals.append(ev)
    finally:
        conn.close()
    return evals


# MARK: - Reporting

def build_report(evals: list[SpineEval], run_json_path: Path, oracle_path: Path, db_path: Path, elapsed_s: float) -> str:
    lines = [
        "# OCR-to-catalog fuzzy match: algorithm comparison",
        "",
        f"Run: `{run_json_path}`  \nOracle: `{oracle_path}`  \nCatalog: `{db_path}`",
        f"\n{len(evals)} oracle-paired, quality-gate-passed spines evaluated in {elapsed_s:.1f}s.",
        "",
        "## Results (accuracy vs. oracle ground truth)",
        "",
        "| Algorithm | Exact hit | Exact+contains hit | No-shortlist (retrieval miss) |",
        "|---|---|---|---|",
    ]
    n = len(evals)
    no_shortlist = sum(1 for e in evals if e.shortlist_size == 0)
    rows = list(ALGORITHMS) + [Algorithm("production (FieldAwareScore)", lambda a, b: 0.0)]
    for algo in rows:
        exact_n = 0
        loose_n = 0
        for e in evals:
            if algo.name == "production (FieldAwareScore)":
                title = e.production_title
            else:
                pick = e.picks.get(algo.name)
                title = pick.title if pick else None
            exact, contains = is_hit(e.oracle_title, title)
            exact_n += int(exact)
            loose_n += int(exact or contains)
        pct_exact = 100.0 * exact_n / n if n else 0.0
        pct_loose = 100.0 * loose_n / n if n else 0.0
        lines.append(f"| {algo.name} | {exact_n}/{n} ({pct_exact:.1f}%) | {loose_n}/{n} ({pct_loose:.1f}%) | {no_shortlist}/{n} |")

    lines += [
        "",
        "## Sample disagreements (production vs. best-performing single-blob algorithm)",
        "",
        "First 15 spines where at least one algorithm's pick differs from the oracle title (exact+contains judge):",
        "",
        "| Spine | OCR text (truncated) | Oracle title | production | token_set_ratio | WRatio | jaro_winkler |",
        "|---|---|---|---|---|---|---|",
    ]
    shown = 0
    for e in evals:
        if shown >= 15:
            break
        picks_for_row = {
            "production": e.production_title,
            "token_set_ratio": (e.picks.get("token_set_ratio").title if e.picks.get("token_set_ratio") else None),
            "WRatio": (e.picks.get("WRatio").title if e.picks.get("WRatio") else None),
            "jaro_winkler": (e.picks.get("jaro_winkler").title if e.picks.get("jaro_winkler") else None),
        }
        any_miss = any(not (is_hit(e.oracle_title, t)[0] or is_hit(e.oracle_title, t)[1]) for t in picks_for_row.values())
        if not any_miss:
            continue
        shown += 1
        ocr_short = e.ocr_text.replace("\n", " / ")[:40]
        lines.append(
            f"| {e.spine_id} | {ocr_short} | {e.oracle_title} | {picks_for_row['production'] or '—'} "
            f"| {picks_for_row['token_set_ratio'] or '—'} | {picks_for_row['WRatio'] or '—'} | {picks_for_row['jaro_winkler'] or '—'} |"
        )
    if shown == 0:
        lines.append("| (none — every algorithm agreed with the oracle on every row) | | | | | | |")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-json", type=Path, default=None, help="Path to a run.json (default: newest under the telemetry dir)")
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--shortlist-cap", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_json_path = args.run_json
    if run_json_path is None:
        runs_dir = DEFAULT_TELEMETRY_DIR / "runs"
        candidates = sorted(runs_dir.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"No run.json found under {runs_dir}", file=sys.stderr)
            return 1
        run_json_path = candidates[0]

    start = time.time()
    evals = run_experiment(run_json_path, args.oracle, args.db, args.shortlist_cap)
    elapsed = time.time() - start
    report = build_report(evals, run_json_path, args.oracle, args.db, elapsed)

    if args.out:
        args.out.write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
