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

Interrupted/error runs resume from the last page flushed to `raw/<list_id>.jsonl`
(scrape-harness reloads that file and continues pagination). Pass `--fresh` with
`--redo-list` only when you intentionally want to discard partial output and
start that list from page 1 again.

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
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
# A chunked run's harness invocation hit policy.max_requests (see
# DEFAULT_MAX_PAGES_PER_RUN) with the list not yet exhausted. Treated like
# `pending` for scheduling (see Checkpoint.pending_lists) but kept distinct so
# --report-only can show mid-list progress separately from never-attempted.
STATUS_CHUNKED = "chunked"

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
# Generous: a ~1000-page list at the harness's per-page delay (2-6s) plus load
# time can run well over an hour; a stuck navigation should still be killed.
HARNESS_TIMEOUT_SECONDS = 4 * 60 * 60

# Chunk long lists into short subprocess sessions rather than one multi-hour
# marathon (list 1's ~790 pages becomes ~16 chunked runs). Mirrors
# list_show.yaml's own policy.max_requests default, but is always passed
# through explicitly via a generated profile (see _write_chunked_profile) so
# this script never has to assume the yaml default stayed in sync.
DEFAULT_MAX_PAGES_PER_RUN = 50
CHUNKED_PROFILE_ID = "list_show_chunked"

# Increasing cooldown before the *same* list is auto-retried again after an
# error/empty_unexpected result, enforced across separate CLI invocations via
# Checkpoint.next_retry_at — not just within one main() call, since manual
# re-runs are exactly how list 1 got hit three times in under an hour.
RETRY_BACKOFF_MINUTES = [10, 45, 120]

# N consecutive plain `error` results (i.e. non-zero exit without a textual
# challenge marker — see _looks_like_challenge) for the same list are treated
# like a challenge: stop the session early rather than keep hammering what
# may be a soft rate-limit/degradation response. Kept conservative to avoid
# an unrelated flaky page load stopping a session over a false positive.
SOFT_BLOCK_THRESHOLD = 3


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


def load_deprecated_list_ids(path: Path) -> set[int]:
    """Reads `seed_list_overrides.yaml` (written by
    `tools/catalog/analyze_goodreads_lists.py --apply-deprecations`) ->
    the set of list_ids marked `curation_status: deprecated`. A missing
    file means nothing is deprecated — never an error, since this file is
    entirely optional and this repo doesn't require the analysis tool to
    have ever been run."""
    if not path.exists():
        return set()
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides") or {}
    return {
        int(list_id)
        for list_id, entry in overrides.items()
        if str((entry or {}).get("curation_status")) == "deprecated"
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _next_retry_at(prior_attempts: int) -> str:
    """Increasing cooldown per RETRY_BACKOFF_MINUTES; caps at the last tier
    rather than growing unboundedly for a list that keeps failing."""
    tier = min(prior_attempts, len(RETRY_BACKOFF_MINUTES) - 1)
    delay = timedelta(minutes=RETRY_BACKOFF_MINUTES[tier])
    return (datetime.now(timezone.utc) + delay).isoformat(timespec="seconds")


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
                updated_at TEXT NOT NULL,
                next_retry_at TEXT,
                consecutive_plain_timeouts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._migrate_columns()
        self.conn.commit()

    def _migrate_columns(self) -> None:
        """Additive migration for checkpoint DBs created before next_retry_at /
        consecutive_plain_timeouts existed — CREATE TABLE IF NOT EXISTS alone
        doesn't add columns to an already-existing table, and this DB holds
        real in-progress scrape state that must not be dropped/recreated."""
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(goodreads_lists)")}
        if "next_retry_at" not in existing:
            self.conn.execute("ALTER TABLE goodreads_lists ADD COLUMN next_retry_at TEXT")
        if "consecutive_plain_timeouts" not in existing:
            self.conn.execute(
                "ALTER TABLE goodreads_lists ADD COLUMN consecutive_plain_timeouts INTEGER NOT NULL DEFAULT 0"
            )

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
        """`chunked` is scheduled alongside `pending`/`error` — it's a list
        mid-way through, not a failure. `next_retry_at` excludes an
        error/empty_unexpected list still cooling down (see record_result)."""
        now = _now_iso()
        cur = self.conn.execute(
            """
            SELECT * FROM goodreads_lists
            WHERE status IN (?, ?, ?) AND attempts < ?
              AND (next_retry_at IS NULL OR next_retry_at <= ?)
            """,
            (STATUS_PENDING, STATUS_ERROR, STATUS_CHUNKED, max_attempts, now),
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
    ) -> sqlite3.Row:
        """Persists one attempt's outcome and returns the updated row.

        `attempts` and `consecutive_plain_timeouts` both reset to 0 on any
        forward-progress result (`done`/`chunked`) rather than accumulating
        forever — they track *consecutive non-progress* attempts, so a list
        that eventually starts succeeding isn't still capped by earlier
        unrelated errors. Only `error`/`empty_unexpected` results set a
        `next_retry_at` cooldown (see RETRY_BACKOFF_MINUTES); `challenged`
        deliberately gets none — it requires a manual --redo-list once a
        human has checked the VPN/session, not an automatic retry.
        """
        now = _now_iso()
        row = self.conn.execute(
            "SELECT attempts, consecutive_plain_timeouts FROM goodreads_lists WHERE list_id = ?",
            (list_id,),
        ).fetchone()
        prior_attempts = row["attempts"] if row else 0
        prior_plain_timeouts = row["consecutive_plain_timeouts"] if row else 0

        if status in (STATUS_DONE, STATUS_CHUNKED):
            attempts = 0
            plain_timeouts = 0
            next_retry_at = None
        else:
            attempts = prior_attempts + 1
            plain_timeouts = prior_plain_timeouts + 1 if status == STATUS_ERROR else 0
            if status in (STATUS_ERROR, STATUS_EMPTY_UNEXPECTED):
                next_retry_at = _next_retry_at(prior_attempts)
            else:
                next_retry_at = None

        self.conn.execute(
            """
            UPDATE goodreads_lists
            SET status = ?, attempts = ?, last_attempt_at = ?, last_error = ?,
                output_path = ?, book_count = ?, updated_at = ?,
                next_retry_at = ?, consecutive_plain_timeouts = ?
            WHERE list_id = ?
            """,
            (status, attempts, now, error, output_path, book_count, now, next_retry_at, plain_timeouts, list_id),
        )
        self.conn.commit()
        return self.conn.execute("SELECT * FROM goodreads_lists WHERE list_id = ?", (list_id,)).fetchone()

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
                book_count = NULL, updated_at = ?, next_retry_at = NULL, consecutive_plain_timeouts = 0
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


def _count_saved_pages(output_path: Path) -> int:
    """One JSONL line per paginated list page (see scrape-harness runtime flush)."""
    if not output_path.exists():
        return 0
    pages = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages += 1
    return pages


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


def _write_chunked_profile(harness_root: Path, max_pages_per_run: int) -> str:
    """Writes/overwrites a `list_show_chunked` profile — a copy of the live
    `list_show` profile with `policy.max_requests` overridden — into the
    harness's own `sites/goodreads/profiles/` directory, and returns its
    profile id.

    This is the "thin profile override file" approach: scrape-harness's
    templating only substitutes `--set key=value` into start_url/url_pattern,
    not into policy config, so a per-run max_requests override has to be a
    real profile file on disk. Rewritten on every call so it always mirrors
    whatever `list_show.yaml` currently does (including future template
    changes, once synced) — never hand-edit the generated file.
    """
    import yaml

    base_path = harness_root / "sites" / "goodreads" / "profiles" / "list_show.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Expected base profile at {base_path} — is the harness site set up?")
    data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    data["id"] = CHUNKED_PROFILE_ID
    policy = dict(data.get("policy") or {})
    policy["max_requests"] = max_pages_per_run
    data["policy"] = policy

    out_path = base_path.parent / f"{CHUNKED_PROFILE_ID}.yaml"
    header = (
        "# AUTO-GENERATED by book-train/tools/scrape_goodreads_lists.py (--max-pages-per-run).\n"
        "# Mirrors list_show.yaml with policy.max_requests overridden. Do not hand-edit —\n"
        "# rewritten on every orchestrator run.\n"
    )
    out_path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return CHUNKED_PROFILE_ID


def run_one_list(
    harness_bin: Path,
    harness_root: Path,
    seed: SeedList,
    out_dir: Path,
    *,
    skip_vpn_check: bool,
    profile_id: str = "list_show",
    max_pages_per_run: int | None = None,
) -> tuple[str, str | None, Path, int | None]:
    """Run `scrape-harness scrape goodreads --profile <profile_id>` for one list.

    `max_pages_per_run` is only used, after a successful run, to tell a
    genuine finish apart from a chunk boundary: if the harness fetched at
    least that many *new* pages this run, the list probably isn't exhausted
    yet even though the process exited 0 (see STATUS_CHUNKED).

    Returns (status, error_message_or_None, output_path, book_count_or_None).
    """
    out_path = out_dir / f"{seed.list_id}.jsonl"
    saved_pages_before = _count_saved_pages(out_path)
    if saved_pages_before:
        print(
            f"[goodreads] list_id={seed.list_id}: resuming from page {saved_pages_before + 1} "
            f"({saved_pages_before} pages already in {out_path})"
        )
    # list_show.yaml has a `list_name` literal field (see its FIELD CONTRACT
    # comment); derive a readable name from slug rather than requiring a
    # separate column in goodreads_seed_lists.yaml.
    list_name = seed.slug.replace("_", " ")
    cmd = [
        str(harness_bin),
        "scrape",
        "goodreads",
        "--profile",
        profile_id,
        "--set",
        f"list_id={seed.list_id}",
        "--set",
        f"slug={seed.slug}",
        "--set",
        f"list_name={list_name}",
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
        partial = _count_books(out_path)
        return status, tail, out_path, partial if partial else None

    book_count = _count_books(out_path)
    if book_count == 0:
        return STATUS_EMPTY_UNEXPECTED, "harness exited 0 but no book_urls were extracted", out_path, 0

    saved_pages_after = _count_saved_pages(out_path)
    pages_fetched = saved_pages_after - saved_pages_before
    if max_pages_per_run is not None and pages_fetched >= max_pages_per_run:
        return STATUS_CHUNKED, None, out_path, book_count
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
        "--skip-deprecated",
        action="store_true",
        help=(
            "Skip list_ids marked curation_status: deprecated in --overrides "
            "(see tools/catalog/analyze_goodreads_lists.py). Off by default — "
            "existing scrape runs are unaffected unless you opt in."
        ),
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="Default: catalog_goodreads('seed_list_overrides.yaml'). Only read when --skip-deprecated is set.",
    )
    parser.add_argument(
        "--redo-list",
        type=int,
        action="append",
        metavar="LIST_ID",
        help="Reset list_id(s) to pending before run (e.g. after raising max_pages). "
        "Keeps existing raw/<id>.jsonl so scrape-harness resumes from the last saved page "
        "unless --fresh is also passed.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="With --redo-list: back up then remove raw/<id>.jsonl so the list is scraped from page 1.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Seed the visit-order shuffle (testing).")
    parser.add_argument(
        "--max-pages-per-run",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_RUN,
        help=(
            "Chunk each list_show harness invocation to at most this many NEW pages "
            "(default: %(default)s) so long lists resume across multiple short sessions "
            "instead of one multi-hour run. Applied via a generated list_show_chunked "
            "profile copy — see _write_chunked_profile."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or catalog_goodreads("raw")
    checkpoint_db = args.checkpoint_db or catalog_goodreads("checkpoint.sqlite")
    checkpoint = Checkpoint(checkpoint_db)

    try:
        seeds = load_seed_lists(args.seed_lists)
        checkpoint.upsert_seed(seeds)

        if args.redo_list:
            for list_id in args.redo_list:
                out_path = out_dir / f"{list_id}.jsonl"
                if args.fresh and out_path.exists():
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    backup = out_path.with_suffix(f".jsonl.bak-{stamp}")
                    shutil.move(out_path, backup)
                    print(f"[goodreads] --fresh: moved {out_path} -> {backup}")
                if checkpoint.reset_list(list_id):
                    print(f"[goodreads] reset list {list_id} to pending")
                else:
                    print(f"[goodreads] warning: list {list_id} not in checkpoint (seed it first)")

        if args.report_only:
            _print_report(checkpoint)
            return 0

        pending = list(checkpoint.pending_lists(args.max_attempts))

        if args.skip_deprecated:
            overrides_path = args.overrides or catalog_goodreads("seed_list_overrides.yaml")
            deprecated_ids = load_deprecated_list_ids(overrides_path)
            if deprecated_ids:
                before = len(pending)
                pending = [row for row in pending if row["list_id"] not in deprecated_ids]
                skipped = before - len(pending)
                if skipped:
                    print(
                        f"[goodreads] --skip-deprecated: skipping {skipped} list(s) "
                        f"marked deprecated (see {overrides_path})"
                    )

        if not pending:
            print("[goodreads] nothing pending (all lists done, or attempts exhausted) — see --report-only")
            return 0

        harness_bin = _harness_binary(args.harness_root)
        profile_id = _write_chunked_profile(args.harness_root, args.max_pages_per_run)

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
                harness_bin,
                args.harness_root,
                seed,
                out_dir,
                skip_vpn_check=args.skip_vpn_check,
                profile_id=profile_id,
                max_pages_per_run=args.max_pages_per_run,
            )
            updated_row = checkpoint.record_result(
                seed.list_id, status=status, error=error, output_path=str(out_path), book_count=book_count
            )
            processed += 1
            suffix = f" ({book_count} books)" if book_count else ""
            print(f"[goodreads] list_id={seed.list_id} -> {status}{suffix}")

            if status == STATUS_CHALLENGED:
                print("[goodreads] challenge detected — stopping session early; check VPN/session before retrying")
                break

            if status == STATUS_ERROR and updated_row["consecutive_plain_timeouts"] >= SOFT_BLOCK_THRESHOLD:
                print(
                    f"[goodreads] list_id={seed.list_id}: "
                    f"{updated_row['consecutive_plain_timeouts']} consecutive plain-timeout errors "
                    "(no challenge marker) — treating like a challenge and stopping session early. "
                    "Consider a cooldown or VPN check before the next --redo-list / auto-retry."
                )
                break

            if processed < len(pending):
                time.sleep(random.uniform(*DEFAULT_INTER_LIST_PAUSE_RANGE))

        _print_report(checkpoint)
        return 0
    finally:
        checkpoint.close()


if __name__ == "__main__":
    raise SystemExit(main())
