#!/usr/bin/env python3
"""Look up spine OCR text via Google Books API and compare API titles to OCR.

Uses a tiered query strategy tuned for noisy vertical-spine OCR blobs:

1. **intitle_phrase** — quoted phrase from the longest run of significant
   tokens (AND semantics inside the phrase; best for coherent title reads).
2. **intitle_tokens** — up to three ``intitle:`` clauses on the longest
   distinctive tokens (AND across tokens; tolerates mashed publisher noise).
3. **fulltext** — cleaned OCR as a general ``q=`` search (fallback).

Each strategy fetches up to ``max_results`` hits; all hits are merged and
reranked with ``token_set_ratio(OCR, title + author)`` so the best semantic
fit wins regardless of which query produced it.

Usage:
    .venv/bin/python3 tools/google_books_ocr_match.py
    .venv/bin/python3 tools/google_books_ocr_match.py --limit 25 --out tools/reports/google_books_ocr_comparison.md
    .venv/bin/python3 tools/test_google_books_ocr_match.py

API key: env ``GOOGLE_BOOKS_API_KEY`` / ``GOOGLE_API_KEY``, else gcloud
``book-train-google-books`` on project ``google-book-api``
(``intricate-idiom-505902-m5``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ocr_parity import OBB, point_in_polygon  # noqa: E402
from spine_matching_parity_fixture import normalize_for_search  # noqa: E402

API_ROOT = "https://www.googleapis.com/books/v1/volumes"
GOOGLE_BOOKS_GCP_PROJECT_ID = "intricate-idiom-505902-m5"
GOOGLE_BOOKS_API_KEY_ID = "book-train-google-books"
DEFAULT_ORACLE = Path(__file__).resolve().parents[2] / "optimize-gemini" / "fixtures" / "oracles" / "bookcase.json"
DEFAULT_TELEMETRY_DIR = Path.home() / "Library" / "Application Support" / "BookID" / "telemetry"
DEFAULT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "google_books_responses.json"


def resolve_google_books_api_key() -> str | None:
    """Env override, else the gcloud key on project google-book-api."""
    env = (os.environ.get("GOOGLE_BOOKS_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if env:
        return env
    try:
        proc = subprocess.run(
            [
                "gcloud",
                "services",
                "api-keys",
                "get-key-string",
                GOOGLE_BOOKS_API_KEY_ID,
                f"--project={GOOGLE_BOOKS_GCP_PROJECT_ID}",
                "--format=value(keyString)",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    key = proc.stdout.strip()
    if key.startswith("keyString:"):
        key = key.split(":", 1)[1].strip()
    return key or None

# Spine/publisher boilerplate — dropped before building intitle queries.
_BOILERPLATE = frozenset(
    """
    edition second third fourth 1st 2nd 3rd 4th revised updated expanded
    pull out pullout foldout fold out map maps pocket handbook guide guides
    lonely planet moon frommers frommer rick steves falcon falconguides
    mountaineers outdoor basics expert advice includes include new
    """.split()
)

_ENGLISH_STOPWORDS = frozenset(
    """
    the a an and or of in on at to for from with by as is are was were be been
    being this that these those it its i you he she they we my your his her their
    our not no but if so than then too very can will just about into over under
    up down out off all any both each few more most other some such only own same
    """.split()
)

_MIN_TOKEN_LEN = 3
_MAX_QUERY_CHARS = 180
_DEFAULT_DELAY_S = 0.35


@dataclass(frozen=True)
class QueryStrategy:
    name: str
    q: str


@dataclass(frozen=True)
class VolumeHit:
    volume_id: str
    title: str
    authors: tuple[str, ...]
    published_date: str
    categories: tuple[str, ...]
    main_category: str
    average_rating: float | None
    ratings_count: int | None
    query_strategy: str
    api_rank: int
    score: float  # token_set_ratio vs OCR blob

    @property
    def author_display(self) -> str:
        return ", ".join(self.authors)

    @property
    def searchable_blob(self) -> str:
        parts = [self.title, self.author_display]
        return normalize_for_search(" ".join(p for p in parts if p))


@dataclass
class LookupResult:
    ocr_text: str
    winning_strategy: str | None
    api_title: str | None
    api_authors: str | None
    volume_id: str | None
    score: float
    ocr_api_fuzzy: float
    api_categories: tuple[str, ...] = ()
    api_main_category: str | None = None
    api_average_rating: float | None = None
    api_ratings_count: int | None = None
    strategies_tried: list[str] = field(default_factory=list)
    top_hits: list[VolumeHit] = field(default_factory=list)
    error: str | None = None


@dataclass
class ComparisonRow:
    spine_id: str
    ocr_text: str
    oracle_title: str
    oracle_author: str
    api_title: str | None
    api_authors: str | None
    winning_strategy: str | None
    ocr_api_fuzzy: float
    api_oracle_fuzzy: float
    ocr_oracle_fuzzy: float
    volume_id: str | None
    api_categories: tuple[str, ...] = ()
    api_main_category: str | None = None
    api_average_rating: float | None = None
    api_ratings_count: int | None = None
    error: str | None = None


def tokenize(normalized: str) -> list[str]:
    return [t for t in normalized.split() if len(t) >= _MIN_TOKEN_LEN]


def significant_tokens(ocr_text: str, *, max_tokens: int = 8) -> list[str]:
    """Distinctive tokens for intitle queries — longest first, stopwords/boilerplate removed."""
    normalized = normalize_for_search(ocr_text)
    seen: set[str] = set()
    candidates: list[str] = []
    for tok in tokenize(normalized):
        if tok in _ENGLISH_STOPWORDS or tok in _BOILERPLATE:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        candidates.append(tok)
    candidates.sort(key=lambda t: (-len(t), t))
    return candidates[:max_tokens]


def longest_token_phrase(ocr_text: str, *, min_tokens: int = 2, max_tokens: int = 6) -> str:
    """Longest contiguous run of significant tokens in OCR reading order."""
    ocr_norm = normalize_for_search(ocr_text)
    sig_set = set(significant_tokens(ocr_text))
    ordered_ocr = [t for t in tokenize(ocr_norm) if t in sig_set]
    if len(ordered_ocr) < min_tokens:
        return ordered_ocr[0] if ordered_ocr else ""

    best = ""
    n = len(ordered_ocr)
    for start in range(n):
        for end in range(start + min_tokens, min(n, start + max_tokens) + 1):
            phrase = " ".join(ordered_ocr[start:end])
            if len(phrase) > len(best):
                best = phrase
    return best


def build_query_strategies(ocr_text: str) -> list[QueryStrategy]:
    """Tiered Google Books queries for one OCR blob."""
    sig = significant_tokens(ocr_text)
    strategies: list[QueryStrategy] = []

    phrase = longest_token_phrase(ocr_text)
    if phrase and len(phrase.split()) >= 2:
        strategies.append(QueryStrategy("intitle_phrase", f'intitle:"{phrase}"'))

    if len(sig) >= 2:
        top = sig[:3]
        intitle_q = " ".join(f"intitle:{t}" for t in top)
        strategies.append(QueryStrategy("intitle_tokens", intitle_q))
    elif len(sig) == 1:
        strategies.append(QueryStrategy("intitle_single", f"intitle:{sig[0]}"))

    cleaned = normalize_for_search(ocr_text)
    if cleaned:
        # General full-text — truncate very long OCR blobs.
        q = cleaned[:_MAX_QUERY_CHARS].strip()
        strategies.append(QueryStrategy("fulltext", q))

    # Dedupe identical q strings while preserving priority order.
    seen_q: set[str] = set()
    out: list[QueryStrategy] = []
    for s in strategies:
        if s.q in seen_q:
            continue
        seen_q.add(s.q)
        out.append(s)
    return out


def _score_hit(ocr_text: str, title: str, authors: tuple[str, ...]) -> float:
    blob = normalize_for_search(f"{title} {' '.join(authors)}".strip())
    ocr_norm = normalize_for_search(ocr_text)
    if not blob or not ocr_norm:
        return 0.0
    return float(fuzz.token_set_ratio(ocr_norm, blob))


def _parse_volume(item: dict, *, query_strategy: str, api_rank: int, ocr_text: str) -> VolumeHit | None:
    vid = item.get("id") or ""
    info = item.get("volumeInfo") or {}
    title = (info.get("title") or "").strip()
    if not title:
        return None
    authors = tuple(a.strip() for a in (info.get("authors") or []) if a and a.strip())
    published = (info.get("publishedDate") or "").strip()
    categories = tuple(c.strip() for c in (info.get("categories") or []) if c and c.strip())
    main_category = (info.get("mainCategory") or "").strip()
    raw_rating = info.get("averageRating")
    average_rating = float(raw_rating) if raw_rating is not None else None
    raw_count = info.get("ratingsCount")
    ratings_count = int(raw_count) if raw_count is not None else None
    score = _score_hit(ocr_text, title, authors)
    return VolumeHit(
        volume_id=vid,
        title=title,
        authors=authors,
        published_date=published,
        categories=categories,
        main_category=main_category,
        average_rating=average_rating,
        ratings_count=ratings_count,
        query_strategy=query_strategy,
        api_rank=api_rank,
        score=score,
    )


def load_fixtures(path: Path) -> dict[str, list[dict]]:
    """Load fixture responses keyed by normalized OCR text."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if "by_ocr" in payload:
        return payload["by_ocr"]
    # Legacy: keyed by query string — not used for replay anymore.
    return {}


def fetch_volumes(
    query: str,
    *,
    api_key: str | None = None,
    max_results: int = 5,
    lang_restrict: str = "en",
    timeout_s: float = 20.0,
    fixtures: dict[str, list[dict]] | None = None,
    fixture_ocr_key: str | None = None,
) -> list[dict]:
    if fixtures is not None and fixture_ocr_key and fixture_ocr_key in fixtures:
        return fixtures[fixture_ocr_key][:max_results]

    params: dict[str, str] = {
        "q": query,
        "maxResults": str(min(max(max_results, 1), 40)),
        "printType": "books",
        "projection": "full",
        "orderBy": "relevance",
    }
    if lang_restrict:
        params["langRestrict"] = lang_restrict
    if api_key:
        params["key"] = api_key
    url = f"{API_ROOT}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("items") or []


def lookup_ocr(
    ocr_text: str,
    *,
    api_key: str | None = None,
    max_results: int = 5,
    delay_s: float = _DEFAULT_DELAY_S,
    strategies: list[QueryStrategy] | None = None,
    fixtures: dict[str, list[dict]] | None = None,
) -> LookupResult:
    ocr_text = (ocr_text or "").strip()
    if not ocr_text:
        return LookupResult(
            ocr_text="",
            winning_strategy=None,
            api_title=None,
            api_authors=None,
            volume_id=None,
            score=0.0,
            ocr_api_fuzzy=0.0,
            error="empty OCR",
        )

    tried: list[str] = []
    merged: dict[str, VolumeHit] = {}
    last_error: str | None = None
    ocr_key = normalize_for_search(ocr_text)
    fixture_items = fixtures.get(ocr_key) if fixtures else None

    strategy_list = (
        [QueryStrategy("fixture", "fixture")]
        if fixture_items is not None
        else (strategies or build_query_strategies(ocr_text))
    )

    for strat in strategy_list:
        tried.append(strat.name)
        try:
            if fixture_items is not None:
                items = fixture_items
            else:
                items = fetch_volumes(
                    strat.q,
                    api_key=api_key,
                    max_results=max_results,
                    fixtures=fixtures,
                    fixture_ocr_key=ocr_key,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:200]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code == 429:
                break
            if delay_s:
                time.sleep(delay_s)
            continue
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
            if delay_s:
                time.sleep(delay_s)
            continue

        for rank, item in enumerate(items):
            hit = _parse_volume(item, query_strategy=strat.name, api_rank=rank, ocr_text=ocr_text)
            if hit is None:
                continue
            existing = merged.get(hit.volume_id)
            if existing is None or hit.score > existing.score:
                merged[hit.volume_id] = hit
        if fixture_items is not None:
            break
        if delay_s:
            time.sleep(delay_s)

    if not merged:
        return LookupResult(
            ocr_text=ocr_text,
            winning_strategy=None,
            api_title=None,
            api_authors=None,
            volume_id=None,
            score=0.0,
            ocr_api_fuzzy=0.0,
            strategies_tried=tried,
            error=last_error or "no results",
        )

    ranked = sorted(merged.values(), key=lambda h: (-h.score, h.api_rank))
    best = ranked[0]
    ocr_api = _score_hit(ocr_text, best.title, best.authors)
    return LookupResult(
        ocr_text=ocr_text,
        winning_strategy=best.query_strategy,
        api_title=best.title,
        api_authors=best.author_display or None,
        volume_id=best.volume_id,
        score=best.score,
        ocr_api_fuzzy=ocr_api,
        api_categories=best.categories,
        api_main_category=best.main_category or None,
        api_average_rating=best.average_rating,
        api_ratings_count=best.ratings_count,
        strategies_tried=tried,
        top_hits=ranked[:5],
    )


def fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(fuzz.token_set_ratio(normalize_for_search(a), normalize_for_search(b)))


def spine_obb(spine: dict) -> OBB:
    return OBB(
        float(spine["cx"]),
        float(spine["cy"]),
        float(spine["w"]),
        float(spine["h"]),
        math.radians(float(spine["angleDeg"])),
    )


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


def load_spines_from_run(run_json: Path) -> list[dict]:
    payload = json.loads(run_json.read_text())
    return payload.get("spines") or []


def compare_run(
    run_json: Path,
    oracle_path: Path,
    *,
    api_key: str | None = None,
    limit: int | None = None,
    delay_s: float = _DEFAULT_DELAY_S,
    max_results: int = 5,
    fixtures: dict[str, list[dict]] | None = None,
) -> list[ComparisonRow]:
    spines = load_spines_from_run(run_json)
    oracle = json.loads(oracle_path.read_text())
    pairing = pair_oracle_indices(oracle, spines)
    rows: list[ComparisonRow] = []

    eligible: list[tuple[int, dict]] = []
    for si, spine in enumerate(spines):
        if si not in pairing:
            continue
        if not spine.get("passedOCRQualityGate", True):
            continue
        ocr = (spine.get("assembledText") or "").strip()
        if not ocr:
            continue
        eligible.append((si, spine))

    if limit is not None:
        eligible = eligible[:limit]

    for si, spine in eligible:
        ocr = spine["assembledText"].strip()
        book = oracle["books"][pairing[si]]
        oracle_title = book.get("title") or ""
        oracle_author = book.get("author") or ""
        lookup = lookup_ocr(
            ocr,
            api_key=api_key,
            delay_s=delay_s,
            max_results=max_results,
            fixtures=fixtures,
        )
        api_title = lookup.api_title
        api_authors = lookup.api_authors
        rows.append(
            ComparisonRow(
                spine_id=spine.get("id", "")[:8],
                ocr_text=ocr,
                oracle_title=oracle_title,
                oracle_author=oracle_author,
                api_title=api_title,
                api_authors=api_authors,
                winning_strategy=lookup.winning_strategy,
                ocr_api_fuzzy=lookup.ocr_api_fuzzy,
                api_oracle_fuzzy=fuzzy(api_title or "", oracle_title),
                ocr_oracle_fuzzy=fuzzy(ocr, oracle_title),
                volume_id=lookup.volume_id,
                api_categories=lookup.api_categories,
                api_main_category=lookup.api_main_category,
                api_average_rating=lookup.api_average_rating,
                api_ratings_count=lookup.api_ratings_count,
                error=lookup.error,
            )
        )
    return rows


def _truncate(s: str, n: int = 56) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_categories(categories: tuple[str, ...]) -> str:
    if not categories:
        return "—"
    return "; ".join(categories)


def _format_rating(rating: float | None, count: int | None) -> str:
    if rating is None and count is None:
        return "—"
    if rating is None:
        return f"n={count}"
    if count is None:
        return f"{rating:.1f}"
    return f"{rating:.1f} ({count})"


def build_markdown_report(
    rows: list[ComparisonRow],
    *,
    run_json: Path,
    oracle_path: Path,
    mode: str = "live",
) -> str:
    n = len(rows)
    with_api = [r for r in rows if r.api_title]
    api_oracle_hits = sum(1 for r in with_api if r.api_oracle_fuzzy >= 70)
    ocr_oracle_hits = sum(1 for r in rows if r.ocr_oracle_fuzzy >= 70)
    api_beats_ocr = sum(
        1 for r in rows if r.api_title and r.api_oracle_fuzzy > r.ocr_oracle_fuzzy + 5
    )
    ocr_beats_api = sum(
        1 for r in rows if r.api_title and r.ocr_oracle_fuzzy > r.api_oracle_fuzzy + 5
    )

    lines = [
        "# Google Books API vs spine OCR (bookcase.jpg)",
        "",
        "> **Live API note:** Without ``GOOGLE_BOOKS_API_KEY``, the shared anonymous",
        "> quota returns HTTP 429. Use ``--fixtures tools/fixtures/google_books_responses.json``",
        "> for offline replay, or set a Books API key and re-run without ``--fixtures``.",
        "",
        f"Run: `{run_json}`  ",
        f"Oracle: `{oracle_path}`  ",
        f"Spines evaluated: **{n}**",
        f"Mode: **{mode}**",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| API returned a title | {len(with_api)}/{n} |",
        f"| API title ↔ oracle fuzzy ≥ 70 | {api_oracle_hits}/{n} |",
        f"| OCR ↔ oracle fuzzy ≥ 70 | {ocr_oracle_hits}/{n} |",
        f"| API clearly better than OCR (+5 fuzzy vs oracle) | {api_beats_ocr}/{n} |",
        f"| OCR clearly better than API (+5 fuzzy vs oracle) | {ocr_beats_api}/{n} |",
        "",
        "## Query strategy",
        "",
        "Tiered queries per spine (first hit pool merged, best ``token_set_ratio`` wins):",
        "",
        "1. ``intitle_phrase`` — quoted longest significant token run from OCR",
        "2. ``intitle_tokens`` — up to 3 ``intitle:`` clauses on longest tokens",
        "3. ``fulltext`` — normalized OCR as general ``q=`` (truncated)",
        "",
        "API params: ``printType=books``, ``langRestrict=en``, ``orderBy=relevance``, ``maxResults=5``, ``projection=full``.",
        "",
        "## Full results",
        "",
        "| Spine | OCR | Oracle title | Google Books title | Author | mainCategory | categories | rating (count) | Strategy | OCR↔API | API↔Oracle | OCR↔Oracle |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r.spine_id} | {r.ocr_text} | {r.oracle_title} | {r.api_title or '—'} "
            f"| {r.api_authors or '—'} | {r.api_main_category or '—'} "
            f"| {_format_categories(r.api_categories)} | "
            f"{_format_rating(r.api_average_rating, r.api_ratings_count)} "
            f"| {r.winning_strategy or '—'} "
            f"| {r.ocr_api_fuzzy:.0f} | {r.api_oracle_fuzzy:.0f} | {r.ocr_oracle_fuzzy:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Notable rows",
            "",
        ]
    )

    def block(title: str, filtered: list[ComparisonRow]) -> None:
        if not filtered:
            return
        lines.append(f"### {title}")
        lines.append("")
        for r in filtered[:8]:
            lines.append(f"- **{r.spine_id}** OCR: `{_truncate(r.ocr_text, 72)}`")
            lines.append(f"  - Oracle: *{r.oracle_title}*")
            lines.append(f"  - API: *{r.api_title or '(none)'}* — {r.api_authors or ''}")
            lines.append(
                f"  - Fuzzy: OCR↔API {r.ocr_api_fuzzy:.0f}, API↔Oracle {r.api_oracle_fuzzy:.0f}, "
                f"OCR↔Oracle {r.ocr_oracle_fuzzy:.0f}"
            )
        lines.append("")

    block(
        "API win (API↔Oracle ≥ 70 and beats OCR by ≥ 5)",
        [r for r in rows if r.api_title and r.api_oracle_fuzzy >= 70 and r.api_oracle_fuzzy >= r.ocr_oracle_fuzzy + 5],
    )
    block(
        "API miss (no result or API↔Oracle < 40)",
        [r for r in rows if not r.api_title or r.api_oracle_fuzzy < 40],
    )
    block(
        "OCR was already close (OCR↔Oracle ≥ 70)",
        [r for r in rows if r.ocr_oracle_fuzzy >= 70],
    )

    return "\n".join(lines) + "\n"


def newest_run_json() -> Path | None:
    runs_dir = DEFAULT_TELEMETRY_DIR / "runs"
    candidates = sorted(runs_dir.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-json", type=Path, default=None)
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--limit", type=int, default=25, help="Max spines to query (default 25; each uses 3 API calls)")
    ap.add_argument("--delay", type=float, default=_DEFAULT_DELAY_S, help="Seconds between API calls")
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="Write markdown report to this path")
    ap.add_argument("--ocr", type=str, default=None, help="Single OCR string smoke test (skips run.json)")
    ap.add_argument("--fixtures", type=Path, default=None, help="Replay Google Books JSON fixtures (offline)")
    ap.add_argument("--record-fixtures", type=Path, default=None, help="Append live API responses to this JSON file")
    args = ap.parse_args()

    api_key = resolve_google_books_api_key()
    fixtures = load_fixtures(args.fixtures) if args.fixtures else None
    mode = f"fixture replay ({args.fixtures})" if fixtures is not None else "live API"

    if args.ocr:
        result = lookup_ocr(
            args.ocr,
            api_key=api_key,
            delay_s=0 if fixtures else args.delay,
            max_results=args.max_results,
            fixtures=fixtures,
        )
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0

    run_json = args.run_json or newest_run_json()
    if run_json is None or not run_json.exists():
        print("No run.json found — pass --run-json or --ocr", file=sys.stderr)
        return 1
    if not args.oracle.exists():
        print(f"Oracle not found: {args.oracle}", file=sys.stderr)
        return 1

    rows = compare_run(
        run_json,
        args.oracle,
        api_key=api_key,
        limit=args.limit,
        delay_s=0 if fixtures else args.delay,
        max_results=args.max_results,
        fixtures=fixtures,
    )
    report = build_markdown_report(rows, run_json=run_json, oracle_path=args.oracle, mode=mode)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
