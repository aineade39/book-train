#!/usr/bin/env python3
"""Pacing benchmark for `book_show_api`: compare the current baseline delay
(3000-8000ms) against a faster tier (1500-4000ms) on a fixed sample of
already-proven-fetchable book IDs, so a live-loop pacing change is backed by
a real before/after comparison instead of a guess after a bad streak.

Bridges `tools/catalog/write_book_show_api_chunked_profile.py`'s
`--delay-ms-min/max` override (a generated `book_show_api_chunked.yaml`) and
`scrape-harness scrape goodreads --set-from-file`: for each tier, writes a
fresh chunked profile, runs one batch against the sample, and tallies
successes / `incomplete_record` warnings / `blocked_suspected` warnings.
Writes only under `--tmp-dir` (default `/tmp`) — never touches the canonical
`book_show_api.jsonl`.

Refuses to run while `run_book_show_api_loop.sh` (or any other scrape) holds
scrape-harness's per-site lock, since racing the live loop for the same VPN
egress would make both runs' numbers meaningless.

Usage:
    python tools/catalog/bench_book_show_api_pacing.py
    python tools/catalog/bench_book_show_api_pacing.py --sample-size 30 --seed 42
"""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.book_show_api_session_health import analyze_chunk  # noqa: E402
from tools.catalog.extract_remaining_ids import default_book_show_api_paths  # noqa: E402
from tools.catalog.write_book_show_api_chunked_profile import write_chunked_profile  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_HARNESS_ROOT = _REPO.parent / "scrape-harness"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 42
SITE_ID = "goodreads"
BASELINE_TIER = ("baseline", 3000, 8000)
FAST_TIER = ("fast", 1500, 4000)
SUCCESS_RATE_TOLERANCE = 0.02


class SiteLockedError(RuntimeError):
    """Raised when the goodreads site lock is already held by another process."""


@dataclass
class TierResult:
    name: str
    delay_ms_min: int
    delay_ms_max: int
    total: int
    successes: int
    incomplete: int
    blocked: int
    elapsed_s: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def books_per_min(self) -> float:
        return (self.total / self.elapsed_s) * 60 if self.elapsed_s else 0.0


def select_sample(book_show_api_paths: Iterable[Path], sample_size: int, seed: int) -> list[int]:
    """Pick up to `sample_size` legacy_ids that have already proven fetchable
    (both `legacy_id` and `isbn13` present), so a bench failure points at the
    delay tier rather than at a book that genuinely lacks an ISBN13.
    Deterministic given the same inputs and `seed`."""
    candidates: set[int] = set()
    for path in book_show_api_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                legacy_id = record.get("legacy_id")
                if legacy_id is None or not record.get("isbn13"):
                    continue
                try:
                    candidates.add(int(legacy_id))
                except (TypeError, ValueError):
                    continue

    population = sorted(candidates)
    if len(population) <= sample_size:
        return population
    return sorted(random.Random(seed).sample(population, sample_size))


def write_sample_file(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in ids) + ("\n" if ids else ""), encoding="utf-8")


def check_site_not_locked(harness_root: Path, site_id: str = SITE_ID) -> None:
    """Non-blocking probe of the exact lock file scrape-harness's own
    `site_lock()` uses (`src/scrape_harness/lock.py`) — fails fast with a
    clear message if it's already held, rather than letting a benchmark run
    silently race the live loop for the same VPN egress."""
    lock_path = harness_root / ".data" / "locks" / f"{site_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = lock_path.open("a+")
    try:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SiteLockedError(
                f"Site lock for '{site_id}' is held (probably run_book_show_api_loop.sh) at "
                f"{lock_path} -- stop the live loop before benchmarking pacing."
            ) from exc
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    finally:
        fd.close()


def parse_tier_output(out_path: Path) -> tuple[int, int, int]:
    """Return `(successes, incomplete, blocked)` from one tier's JSONL.

    Same success rule as live spike detection (`analyze_chunk`): isbn13 and
    legacy_id both present. A legacy_id-only row is a silent_null failure,
    not a success — otherwise the bench can promote a delay the live circuit
    breaker would still trip.
    """
    stats = analyze_chunk(out_path, since_line=0)
    return (
        stats.successes,
        stats.warning_types.get("incomplete_record", 0),
        stats.warning_types.get("blocked_suspected", 0),
    )


def run_tier(
    harness_root: Path,
    name: str,
    delay_ms_min: int,
    delay_ms_max: int,
    sample_path: Path,
    sample_size: int,
    out_path: Path,
) -> TierResult:
    if out_path.exists():
        out_path.unlink()
    write_chunked_profile(harness_root, max_requests=sample_size, delay_ms_min=delay_ms_min, delay_ms_max=delay_ms_max)

    print(f"[bench_book_show_api_pacing] running {name} tier ({delay_ms_min}-{delay_ms_max}ms) on {sample_size} books")
    started = time.monotonic()
    result = subprocess.run(
        [
            ".venv/bin/scrape-harness",
            "scrape",
            "goodreads",
            "--profile",
            "book_show_api_chunked",
            "--set-from-file",
            f"book_id={sample_path}",
            "--out",
            str(out_path),
        ],
        cwd=harness_root,
    )
    elapsed_s = time.monotonic() - started
    if result.returncode != 0:
        print(f"[bench_book_show_api_pacing] {name} tier exited {result.returncode} -- tallying what was written anyway")

    successes, incomplete, blocked = parse_tier_output(out_path)
    return TierResult(
        name=name,
        delay_ms_min=delay_ms_min,
        delay_ms_max=delay_ms_max,
        total=sample_size,
        successes=successes,
        incomplete=incomplete,
        blocked=blocked,
        elapsed_s=elapsed_s,
    )


def verdict(baseline: TierResult, fast: TierResult, tolerance: float = SUCCESS_RATE_TOLERANCE) -> bool:
    """Promote the fast tier only if it matches baseline's success rate
    (within `tolerance`) and never triggered a bot-wall block."""
    if fast.blocked > 0:
        return False
    return fast.success_rate >= baseline.success_rate - tolerance


def print_report(baseline: TierResult, fast: TierResult) -> None:
    for tier in (baseline, fast):
        print(
            f"[bench_book_show_api_pacing] {tier.name:>8} {tier.delay_ms_min}-{tier.delay_ms_max}ms: "
            f"{tier.successes}/{tier.total} success ({tier.success_rate:.0%}), "
            f"{tier.incomplete} incomplete, {tier.blocked} blocked, "
            f"{tier.elapsed_s:.0f}s ({tier.books_per_min:.1f} books/min)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS_ROOT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tmp-dir", type=Path, default=Path("/tmp"), help="Scratch dir for the sample + tier output (default: /tmp)."
    )
    args = parser.parse_args(argv)

    try:
        check_site_not_locked(args.harness_root)
    except SiteLockedError as exc:
        print(f"[bench_book_show_api_pacing] {exc}", file=sys.stderr)
        return 1

    book_show_api_paths = default_book_show_api_paths(catalog_goodreads())
    sample = select_sample(book_show_api_paths, args.sample_size, args.seed)
    if not sample:
        print(
            "[bench_book_show_api_pacing] no known-good book IDs found -- run book_show_api at least once first",
            file=sys.stderr,
        )
        return 1

    sample_path = args.tmp_dir / "book_show_api_pacing_sample.txt"
    write_sample_file(sample_path, sample)
    print(f"[bench_book_show_api_pacing] sampled {len(sample)} known-good book IDs (seed={args.seed}) -> {sample_path}")

    baseline_name, baseline_min, baseline_max = BASELINE_TIER
    fast_name, fast_min, fast_max = FAST_TIER

    baseline = run_tier(
        args.harness_root,
        baseline_name,
        baseline_min,
        baseline_max,
        sample_path,
        len(sample),
        args.tmp_dir / "book_show_api_pacing_baseline.jsonl",
    )
    fast = run_tier(
        args.harness_root,
        fast_name,
        fast_min,
        fast_max,
        sample_path,
        len(sample),
        args.tmp_dir / "book_show_api_pacing_fast.jsonl",
    )

    print_report(baseline, fast)

    if verdict(baseline, fast):
        print(
            f"[bench_book_show_api_pacing] PROMOTE: fast tier ({fast_min}-{fast_max}ms) matches baseline -- "
            f"safe to run the loop with DELAY_MS_MIN={fast_min} DELAY_MS_MAX={fast_max}"
        )
        return 0

    print(
        f"[bench_book_show_api_pacing] DO NOT PROMOTE: fast tier ({fast_min}-{fast_max}ms) underperformed "
        "baseline or hit a bot-wall -- keep the 3000-8000ms defaults"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
