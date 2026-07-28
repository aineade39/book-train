#!/usr/bin/env python3
"""Goodreads Listopia scrape orchestrator: checkpointed, restartable, paced.

Drives `sites/goodreads/profiles/list_show.yaml` in a sibling scrape-harness
checkout, once per list in `goodreads_seed_lists.yaml` (append more there —
by hand, or from `list_discovery` output — as needed). All progress lives in
a small SQLite checkpoint DB under `tools.paths.catalog_goodreads()`, NOT in
this repo (see DATA.md) — re-running this script after a crash, a VPN drop,
or just tomorrow picks up where it left off; lists already marked `done` are
skipped, and a `challenged` result stops the whole run early rather than
plowing through the rest of the session into what's likely a wider block.

This is intentionally a low-throughput, infrequent job: a session cap (list
count and wall-clock minutes) plus inter-list pacing jitter keep any single
run short, rather than blasting through every seed list back-to-back.

Requires PyYAML (already a repo dependency; see other tools/catalog/*.py).

Usage:
    python tools/scrape_goodreads_lists.py
    python tools/scrape_goodreads_lists.py --report-only
    python tools/scrape_goodreads_lists.py --harness-root ~/dev/scrape-harness
    python tools/scrape_goodreads_lists.py --session-max-lists 3
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_HARNESS_ROOT = _REPO.parent / "scrape-harness"
SEED_LISTS_PATH = Path(__file__).resolve().parent / "catalog" / "goodreads_seed_lists.yaml"

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_CHALLENGED = "challenged"
STATUS_EMPTY_UNEXPECTED = "empty_unexpected"
STATUS_ERROR = "error"

# Heuristic only: if the harness process exits non-zero and its combined
# stdout/stderr contains one of these, we treat the outcome as `challenged`
# rather than a plain `error`. `challenged` results are never auto-retried
# within a session (see main()) — retrying straight into a block is more
# likely to widen it than clear it; a human should check the VPN/session
# first. Extend this list if a real run turns up a marker missed here.
_CHALLENGE_MARKERS = (
    "captcha",
    "unusual traffic",
    "pardon our interruption",
    "access denied",
    "are you a robot",
    "blocked",
    "verify you are a human",
    "rate limit",
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_SESSION_MAX_LISTS = 6
DEFAULT_SESSION_MAX_MINUTES = 45.0
DEFAULT_INTER_LIST_PAUSE_RANGE = (8.0, 25.0)
# Generous: a ~1000-page list at the harness's per-page delay (1.5-4s) plus load
# time can run well over an hour; a stuck navigation should still be killed.
HARNESS_TIMEOUT_SECONDS = 4 * 60 * 60


@dataclass(frozen=True)
class SeedList:
    list_id: int
    slug: str
    genre: str


def load_seed_lists(path: Path) -> list[SeedList]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        SeedList(list_id=int(row["list_id"]), slug=str(row["slug"]), genre=str(row.get("genre", "")))
        for row in data["lists"]
    ]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Checkpoint:
    """SQLite-backed progress tracker — see module docstring for the restart contract."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goodreads_lists (
                list_id INTEGER PRIMARY KEY,
                slug TEXT NOT NULL,
                genre TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT,
                last_error TEXT,
                output_path TEXT,
                book_count INTEGER,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert_seed(self, seeds: Iterable[SeedList]) -> None:
        """Insert new lists as `pending`; refresh slug/genre for existing ones without
        touching their status/attempts, so a list already marked `done` stays `done`
        even if its slug changed slightly upstream."""
        now = _now_iso()
        for s in seeds:
            self.conn.execute(
                """
                INSERT INTO goodreads_lists (list_id, slug, genre, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(list_id) DO UPDATE SET slug = excluded.slug, genre = excluded.genre
                """,
                (s.list_id, s.slug, s.genre, STATUS_PENDING, now),
            )
        self.conn.commit()

    def pending_lists(self, max_attempts: int) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM goodreads_lists WHERE status IN (?, ?) AND attempts < ?",
            (STATUS_PENDING, STATUS_ERROR, max_attempts),
        )
        return cur.fetchall()

    def record_result(
        self,
        list_id: int,
        *,
        status: str,
        error: str | None,
        output_path: str | None,
        book_count: int | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE goodreads_lists
            SET status = ?, attempts = attempts + 1, last_attempt_at = ?, last_error = ?,
                output_path = ?, book_count = ?, updated_at = ?
            WHERE list_id = ?
            """,
            (status, _now_iso(), error, output_path, book_count, _now_iso(), list_id),
        )
        self.conn.commit()

    def summary(self) -> dict[str, int]:
        cur = self.conn.execute("SELECT status, COUNT(*) AS n FROM goodreads_lists GROUP BY status")
        return {row["status"]: row["n"] for row in cur.fetchall()}

    def rows(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM goodreads_lists ORDER BY list_id").fetchall()

    def reset_list(self, list_id: int) -> bool:
        """Mark a list pending again so it will be re-scraped (e.g. after raising max_pages)."""
        cur = self.conn.execute(
            """
            UPDATE goodreads_lists
            SET status = ?, attempts = 0, last_error = NULL, output_path = NULL,
                book_count = NULL, updated_at = ?
            WHERE list_id = ?
            """,
            (STATUS_PENDING, _now_iso(), list_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self.conn.close()


def _harness_binary(harness_root: Path) -> Path:
    candidate = harness_root / ".venv" / "bin" / "scrape-harness"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find scrape-harness console script at {candidate}. "
            f"Run `uv sync` in {harness_root} first, or pass --harness-root."
        )
    return candidate


def _count_books(output_path: Path) -> int:
    """Sum `len(book_urls)` across every page-record in the profile's JSONL output."""
    if not output_path.exists():
        return 0
    total = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += len(record.get("book_urls") or [])
    return total


def _looks_like_challenge(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


def run_one_list(
    harness_bin: Path,
    harness_root: Path,
    seed: SeedList,
    out_dir: Path,
    *,
    skip_vpn_check: bool,
) -> tuple[str, str | None, Path, int | None]:
    """Run `scrape-harness scrape goodreads --profile list_show` for one list.

    Returns (status, error_message_or_None, output_path, book_count_or_None).
    """
    out_path = out_dir / f"{seed.list_id}.jsonl"
    cmd = [
        str(harness_bin),
        "scrape",
        "goodreads",
        "--profile",
        "list_show",
        "--set",
        f"list_id={seed.list_id}",
        "--set",
        f"slug={seed.slug}",
        "--out",
        str(out_path),
    ]
    if skip_vpn_check:
        cmd.append("--skip-vpn-check")

    print(f"[goodreads] list_id={seed.list_id} slug={seed.slug} -> {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=harness_root,
            capture_output=True,
            text=True,
            timeout=HARNESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return STATUS_ERROR, f"harness timed out after {HARNESS_TIMEOUT_SECONDS}s: {exc}", out_path, None

    if proc.returncode != 0:
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        status = STATUS_CHALLENGED if _looks_like_challenge(combined) else STATUS_ERROR
        tail = combined.strip()[-2000:] or f"harness exited {proc.returncode} with no output"
        return status, tail, out_path, None

    book_count = _count_books(out_path)
    if book_count == 0:
        return STATUS_EMPTY_UNEXPECTED, "harness exited 0 but no book_urls were extracted", out_path, 0
    return STATUS_DONE, None, out_path, book_count


def _print_report(checkpoint: Checkpoint) -> None:
    print("[goodreads] checkpoint summary:", json.dumps(checkpoint.summary(), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS_ROOT, help="scrape-harness checkout")
    parser.add_argument("--seed-lists", type=Path, default=SEED_LISTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: catalog_goodreads('raw')")
    parser.add_argument(
        "--checkpoint-db", type=Path, default=None, help="Default: catalog_goodreads('checkpoint.sqlite')"
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--session-max-lists", type=int, default=DEFAULT_SESSION_MAX_LISTS)
    parser.add_argument("--session-max-minutes", type=float, default=DEFAULT_SESSION_MAX_MINUTES)
    parser.add_argument(
        "--skip-vpn-check", action="store_true", help="Forwarded to the harness. Testing only — do not use for real scrapes."
    )
    parser.add_argument("--report-only", action="store_true", help="Print checkpoint status and exit; no scraping.")
    parser.add_argument(
        "--redo-list",
        type=int,
        action="append",
        metavar="LIST_ID",
        help="Reset list_id(s) to pending before run (e.g. after raising max_pages).",
    )
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Seed the visit-order shuffle (testing).")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or catalog_goodreads("raw")
    checkpoint_db = args.checkpoint_db or catalog_goodreads("checkpoint.sqlite")
    checkpoint = Checkpoint(checkpoint_db)

    try:
        seeds = load_seed_lists(args.seed_lists)
        checkpoint.upsert_seed(seeds)

        if args.redo_list:
            for list_id in args.redo_list:
                if checkpoint.reset_list(list_id):
                    print(f"[goodreads] reset list {list_id} to pending")
                else:
                    print(f"[goodreads] warning: list {list_id} not in checkpoint (seed it first)")

        if args.report_only:
            _print_report(checkpoint)
            return 0

        pending = list(checkpoint.pending_lists(args.max_attempts))
        if not pending:
            print("[goodreads] nothing pending (all lists done, or attempts exhausted) — see --report-only")
            return 0

        harness_bin = _harness_binary(args.harness_root)

        # Shuffled so a run never re-visits lists in the same fixed order every
        # time — one of several small behavioral tells (see plan notes on
        # session pacing) that's free to eliminate.
        random.Random(args.shuffle_seed).shuffle(pending)

        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        processed = 0

        for row in pending:
            if processed >= args.session_max_lists:
                print(f"[goodreads] session cap reached ({args.session_max_lists} lists) — stopping for now")
                break
            elapsed_minutes = (time.monotonic() - start) / 60
            if processed > 0 and elapsed_minutes >= args.session_max_minutes:
                print(f"[goodreads] session time cap reached ({args.session_max_minutes} min) — stopping for now")
                break

            seed = SeedList(list_id=row["list_id"], slug=row["slug"], genre=row["genre"] or "")
            status, error, out_path, book_count = run_one_list(
                harness_bin, args.harness_root, seed, out_dir, skip_vpn_check=args.skip_vpn_check
            )
            checkpoint.record_result(
                seed.list_id, status=status, error=error, output_path=str(out_path), book_count=book_count
            )
            processed += 1
            suffix = f" ({book_count} books)" if book_count else ""
            print(f"[goodreads] list_id={seed.list_id} -> {status}{suffix}")

            if status == STATUS_CHALLENGED:
                print("[goodreads] challenge detected — stopping session early; check VPN/session before retrying")
                break

            if processed < len(pending):
                time.sleep(random.uniform(*DEFAULT_INTER_LIST_PAUSE_RANGE))

        _print_report(checkpoint)
        return 0
    finally:
        checkpoint.close()


if __name__ == "__main__":
    raise SystemExit(main())
