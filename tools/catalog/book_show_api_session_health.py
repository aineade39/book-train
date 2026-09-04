#!/usr/bin/env python3
"""Detect book_show_api scrape failure spikes, classify the cause, and recover.

Used by `tools/run_book_show_api_loop.sh` after each harness chunk: if the
session's new JSONL rows are mostly warnings, classify as catalog /
stale_build / soft_block / hard_block, then repair the queue, rediscover the
Next.js build id, escalate a soft-block cooldown, or rotate Mullvad /
hard-stop for a VPN change.

Usage:
    python tools/catalog/book_show_api_session_health.py check \\
        --jsonl path/to/book_show_api.jsonl --since-line 7000

    python tools/catalog/book_show_api_session_health.py recover \\
        --jsonl path/to/book_show_api.jsonl --since-line 7000 \\
        --harness-root ~/dev/scrape-harness

    python tools/catalog/book_show_api_session_health.py refresh-build \\
        --harness-root ~/dev/scrape-harness

    python tools/catalog/book_show_api_exit_session.py prepare-exit \\
        --harness-root ~/dev/scrape-harness

    python tools/catalog/book_show_api_session_health.py reset-state
    python tools/catalog/book_show_api_session_health.py repair
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.extract_remaining_ids import (  # noqa: E402
    DEFAULT_FETCHED_SIDECAR_NAME,
    DEFAULT_GAVE_UP_SIDECAR_NAME,
    DEFAULT_POPULARITY_SIDECAR_NAME,
    DEFAULT_RETRY_AFTER_SIDECAR_NAME,
    build_remaining_queue,
    default_book_show_api_paths,
    load_popularity_sidecar,
    write_ids_remaining,
)
from tools.catalog.list_show_popularity import BookPopularity  # noqa: E402
from tools.catalog.mullvad_rotate import maybe_rotate_exit  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_MIN_CHUNK_RECORDS = 20
DEFAULT_FAILURE_SPIKE_THRESHOLD = 0.80
DEFAULT_TRAILING_WARN_ABORT = 8
DEFAULT_QUEUE_TOP_N = 10
# If this many of the remaining pool have ratings but the queue head does not,
# ordering is almost certainly broken (e.g. empty book_popularity.json sidecar).
DEFAULT_QUEUE_RATINGS_FRACTION = 0.10
DEFAULT_HARD_BLOCK_FRACTION = 0.50
DEFAULT_HARNESS_ROOT = _REPO.parent / "scrape-harness"
DEFAULT_STATE_NAME = "spike_recovery_state.json"
DEFAULT_PROBE_BOOK_ID = 33
# 15m → 45m → 2h after in-session backoff + short stale-build retries are exhausted.
DEFAULT_SPIKE_COOLDOWN_TIERS = (900, 2700, 7200)
DEFAULT_SPIKE_MAX_TIERS = 3
DEFAULT_CATALOG_COOLDOWN_SECONDS = 300
# Cheap rediscover retries before falling through to the 15m soft-block ladder.
# A 404 json_parse streak is a stale build id, not an IP throttle — jumping
# straight to 15m/45m/2h after one failed probe is what turned a minutes-long
# rediscover into hours of downtime.
DEFAULT_STALE_BUILD_RETRY_COOLDOWNS = (60, 180)
# After a successful Mullvad rotate, don't sit 15m–2h on a fresh exit.
POST_ROTATE_COOLDOWN_FLOOR = 60
POST_ROTATE_COOLDOWN_CAP = 180
WARNING_BLOCKED = "blocked_suspected"
WARNING_INCOMPLETE = "incomplete_record"
WARNING_JSON = "json_parse_error"
# Recover exit codes for the orchestrator loop.
RECOVER_OK = 0
RECOVER_ESCALATED = 1
RECOVER_HARD_STOP = 2


class SpikeKind(str, Enum):
    CATALOG = "catalog"
    STALE_BUILD = "stale_build"
    SOFT_BLOCK = "soft_block"
    HARD_BLOCK = "hard_block"


@dataclass(frozen=True)
class ChunkStats:
    total: int
    successes: int
    warnings: int
    warning_types: Counter[str] = field(default_factory=Counter)
    status_codes: Counter[int] = field(default_factory=Counter)
    trailing_warnings: int = 0
    trailing_warning_types: Counter[str] = field(default_factory=Counter)
    trailing_status_codes: Counter[int] = field(default_factory=Counter)

    @property
    def warning_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.warnings / self.total


@dataclass(frozen=True)
class Diagnosis:
    issues: tuple[str, ...]
    repairs: tuple[str, ...]


@dataclass
class RecoveryState:
    tier: int = 0
    consecutive_spikes: int = 0
    last_spike_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "consecutive_spikes": self.consecutive_spikes,
            "last_spike_at": self.last_spike_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryState:
        return cls(
            tier=int(data.get("tier") or 0),
            consecutive_spikes=int(data.get("consecutive_spikes") or 0),
            last_spike_at=data.get("last_spike_at"),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def analyze_chunk(jsonl_path: Path, since_line: int) -> ChunkStats:
    """Count successes vs warnings among JSONL lines written after `since_line`."""
    if not jsonl_path.exists():
        return ChunkStats(total=0, successes=0, warnings=0)

    successes = warnings = 0
    warning_types: Counter[str] = Counter()
    status_codes: Counter[int] = Counter()
    trailing_warnings = 0
    trailing_warning_types: Counter[str] = Counter()
    trailing_status_codes: Counter[int] = Counter()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < since_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = record.get("_status")
            if isinstance(status, int):
                status_codes[status] += 1
            if record.get("_scrape_warning"):
                warnings += 1
                warning_types[str(record["_scrape_warning"])] += 1
                trailing_warnings += 1
                trailing_warning_types[str(record["_scrape_warning"])] += 1
                if isinstance(status, int):
                    trailing_status_codes[status] += 1
            elif record.get("isbn13") and record.get("legacy_id"):
                successes += 1
                trailing_warnings = 0
                trailing_warning_types = Counter()
                trailing_status_codes = Counter()
            else:
                warnings += 1
                warning_types["silent_null"] += 1
                trailing_warnings += 1
                trailing_warning_types["silent_null"] += 1
                if isinstance(status, int):
                    trailing_status_codes[status] += 1

    return ChunkStats(
        total=successes + warnings,
        successes=successes,
        warnings=warnings,
        warning_types=warning_types,
        status_codes=status_codes,
        trailing_warnings=trailing_warnings,
        trailing_warning_types=trailing_warning_types,
        trailing_status_codes=trailing_status_codes,
    )


def is_failure_spike(
    stats: ChunkStats,
    *,
    min_records: int = DEFAULT_MIN_CHUNK_RECORDS,
    threshold: float = DEFAULT_FAILURE_SPIKE_THRESHOLD,
    trailing_abort: int = DEFAULT_TRAILING_WARN_ABORT,
) -> bool:
    """True when the chunk is mostly warnings, or the session ended on a warn streak.

    The harness circuit-breaks after 8 consecutive 404s, so a chunk can be 100
    successes + 8 trailing json_parse_error. Overall rate would miss that; the
    trailing run is the signal that delay cannot recover.
    """
    if stats.trailing_warnings >= trailing_abort:
        return True
    if stats.total < min_records:
        return False
    return stats.warning_rate >= threshold


def popularity_sidecar_is_corrupt(sidecar_path: Path, raw_dir: Path) -> bool:
    raw_paths = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    if not raw_paths:
        return False
    return not load_popularity_sidecar(sidecar_path)


def queue_ordering_suspect(
    ordered_remaining: list[int],
    popularity: dict[int, BookPopularity],
    remaining: set[int],
    *,
    top_n: int = DEFAULT_QUEUE_TOP_N,
    ratings_fraction: float = DEFAULT_QUEUE_RATINGS_FRACTION,
) -> bool:
    if len(ordered_remaining) < top_n or not remaining:
        return False
    head = ordered_remaining[:top_n]
    if any(popularity.get(book_id, BookPopularity()).ratings_count > 0 for book_id in head):
        return False
    with_ratings = sum(1 for book_id in remaining if popularity.get(book_id, BookPopularity()).ratings_count > 0)
    return with_ratings >= len(remaining) * ratings_fraction


def _queue_paths(catalog: Path) -> dict[str, Path]:
    return {
        "raw_dir": catalog / "raw",
        "fetched_sidecar": catalog / DEFAULT_FETCHED_SIDECAR_NAME,
        "gave_up_sidecar": catalog / DEFAULT_GAVE_UP_SIDECAR_NAME,
        "popularity_sidecar": catalog / DEFAULT_POPULARITY_SIDECAR_NAME,
        "retry_after_sidecar": catalog / DEFAULT_RETRY_AFTER_SIDECAR_NAME,
    }


def diagnose_catalog(catalog_dir: Path | None = None) -> Diagnosis:
    catalog = Path(catalog_dir or catalog_goodreads(""))
    paths = _queue_paths(catalog)
    popularity_sidecar = paths["popularity_sidecar"]
    ids_remaining = catalog / "ids_remaining.txt"

    issues: list[str] = []
    repairs: list[str] = []

    if popularity_sidecar_is_corrupt(popularity_sidecar, paths["raw_dir"]):
        issues.append(f"popularity sidecar empty or missing entries ({popularity_sidecar})")
        repairs.append(f"delete {popularity_sidecar.name} and rebuild from raw/")

    queue = build_remaining_queue(
        paths["raw_dir"],
        default_book_show_api_paths(catalog),
        fetched_sidecar=paths["fetched_sidecar"],
        gave_up_sidecar=paths["gave_up_sidecar"],
        popularity_sidecar=popularity_sidecar,
        retry_after_sidecar=paths["retry_after_sidecar"],
        persist=False,
    )

    if queue_ordering_suspect(queue.ordered, queue.popularity, queue.remaining):
        issues.append(
            f"ids_remaining.txt head has no ratings_count signal "
            f"(top book_id={queue.ordered[0] if queue.ordered else 'n/a'}) despite "
            f"{sum(1 for b in queue.remaining if queue.popularity.get(b, BookPopularity()).ratings_count > 0):,} "
            f"remaining books with ratings"
        )
        if f"delete {popularity_sidecar.name}" not in " ".join(repairs):
            repairs.append(f"rebuild {ids_remaining.name} with popularity ordering")

    return Diagnosis(issues=tuple(issues), repairs=tuple(repairs))


def repair_catalog(catalog_dir: Path | None = None) -> tuple[bool, Diagnosis]:
    """Apply safe auto-fixes. Returns (repaired, diagnosis)."""
    catalog = Path(catalog_dir or catalog_goodreads(""))
    diagnosis = diagnose_catalog(catalog)
    if not diagnosis.issues:
        return False, diagnosis

    paths = _queue_paths(catalog)
    popularity_sidecar = paths["popularity_sidecar"]
    ids_remaining = catalog / "ids_remaining.txt"

    if popularity_sidecar_is_corrupt(popularity_sidecar, paths["raw_dir"]):
        if popularity_sidecar.exists():
            popularity_sidecar.unlink()

    queue = build_remaining_queue(
        paths["raw_dir"],
        default_book_show_api_paths(catalog),
        fetched_sidecar=paths["fetched_sidecar"],
        gave_up_sidecar=paths["gave_up_sidecar"],
        popularity_sidecar=popularity_sidecar,
        retry_after_sidecar=paths["retry_after_sidecar"],
        persist=True,
    )
    write_ids_remaining(ids_remaining, queue.ordered)

    return True, diagnosis


def classify_chunk(
    jsonl_path: Path,
    since_line: int,
    *,
    catalog_dir: Path | None = None,
    stats: ChunkStats | None = None,
    hard_block_fraction: float = DEFAULT_HARD_BLOCK_FRACTION,
    trailing_abort: int = DEFAULT_TRAILING_WARN_ABORT,
) -> SpikeKind:
    """Classify a failure-spike chunk.

    Hard-block (majority `blocked_suspected`) wins over catalog repair. Catalog
    issues win over generic incomplete_record. A majority of `json_parse_error`
    warnings or HTTP 404 records is a stale Next.js build id, not an IP throttle.
    When the harness aborted on a trailing warn streak, classify from that tail
    so 100 successes + 8 trailing 404s still count as stale_build.
    """
    stats = stats or analyze_chunk(jsonl_path, since_line)
    if stats.trailing_warnings >= trailing_abort:
        # Trailing 404/bot-wall is a dead session, not a queue bug. Trailing
        # incomplete_record still goes through catalog diagnosis — that is
        # how an empty popularity sidecar presents.
        n = stats.trailing_warnings
        skip_catalog = bool(
            (stats.trailing_warning_types.get(WARNING_JSON, 0) / n >= hard_block_fraction)
            or (stats.trailing_warning_types.get(WARNING_BLOCKED, 0) / n >= hard_block_fraction)
            or (stats.trailing_status_codes.get(404, 0) / n >= hard_block_fraction)
        )
        return _classify_from_counts(
            warnings=stats.trailing_warnings,
            warning_types=stats.trailing_warning_types,
            status_codes=stats.trailing_status_codes,
            total=stats.trailing_warnings,
            catalog_dir=catalog_dir,
            hard_block_fraction=hard_block_fraction,
            skip_catalog=skip_catalog,
        )
    return _classify_from_counts(
        warnings=stats.warnings,
        warning_types=stats.warning_types,
        status_codes=stats.status_codes,
        total=stats.total,
        catalog_dir=catalog_dir,
        hard_block_fraction=hard_block_fraction,
        skip_catalog=False,
    )


def _classify_from_counts(
    *,
    warnings: int,
    warning_types: Counter[str],
    status_codes: Counter[int],
    total: int,
    catalog_dir: Path | None,
    hard_block_fraction: float,
    skip_catalog: bool,
) -> SpikeKind:
    blocked = warning_types.get(WARNING_BLOCKED, 0)
    if warnings and blocked / warnings >= hard_block_fraction:
        return SpikeKind.HARD_BLOCK
    if not skip_catalog:
        diagnosis = diagnose_catalog(catalog_dir)
        if diagnosis.issues:
            return SpikeKind.CATALOG
    json_parse = warning_types.get(WARNING_JSON, 0)
    status_404 = status_codes.get(404, 0)
    if warnings and json_parse / warnings >= hard_block_fraction:
        return SpikeKind.STALE_BUILD
    if total and status_404 / total >= hard_block_fraction:
        return SpikeKind.STALE_BUILD
    return SpikeKind.SOFT_BLOCK


def default_state_path(catalog_dir: Path | None = None) -> Path:
    if catalog_dir is not None:
        return Path(catalog_dir) / DEFAULT_STATE_NAME
    return catalog_goodreads(DEFAULT_STATE_NAME)


def load_recovery_state(path: Path) -> RecoveryState:
    if not path.exists():
        return RecoveryState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RecoveryState()
    if not isinstance(data, dict):
        return RecoveryState()
    return RecoveryState.from_dict(data)


def save_recovery_state(path: Path, state: RecoveryState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")


def reset_recovery_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def pick_probe_book_id(jsonl_path: Path, catalog_dir: Path | None = None) -> int:
    """Prefer the most recent successful isbn13+legacy_id row, else fetched_ids, else 33."""
    if jsonl_path.exists():
        last_ok: int | None = None
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("_scrape_warning"):
                    continue
                legacy_id = record.get("legacy_id")
                if record.get("isbn13") and legacy_id is not None:
                    try:
                        last_ok = int(legacy_id)
                    except (TypeError, ValueError):
                        continue
        if last_ok is not None:
            return last_ok

    catalog = Path(catalog_dir) if catalog_dir is not None else Path(catalog_goodreads(""))
    sidecar = catalog / DEFAULT_FETCHED_SIDECAR_NAME
    if sidecar.exists():
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                return int(line)
            except ValueError:
                continue
    return DEFAULT_PROBE_BOOK_ID


def _harness_binary(harness_root: Path) -> Path:
    return harness_root / ".venv" / "bin" / "scrape-harness"


def wipe_chunked_profile(harness_root: Path) -> None:
    chunked = harness_root / ".data" / "browser_profiles" / "goodreads" / "book_show_api_chunked"
    if chunked.exists():
        shutil.rmtree(chunked, ignore_errors=True)
        print(f"[session_health] wiped browser profile {chunked}")
    lock_path = harness_root / ".data" / "locks" / "goodreads.lock"
    if lock_path.exists():
        try:
            lock_path.unlink()
            print(f"[session_health] cleared stale site lock {lock_path}")
        except OSError as exc:
            print(f"[session_health] could not clear site lock {lock_path}: {exc}")


def refresh_next_build(harness_root: Path, *, timeout: int = 90) -> bool:
    from tools.catalog.book_show_api_exit_session import prepare_exit

    print("[session_health] refreshing Next.js build id (home → list → book, shared chunked profile)")
    return prepare_exit(harness_root, fresh_browser=True, timeout=timeout)


def probe_known_good_book(
    harness_root: Path,
    book_id: int,
    out_path: Path,
    *,
    timeout: int = 180,
) -> bool:
    harness_bin = _harness_binary(harness_root)
    if not harness_bin.exists():
        print(f"[session_health] scrape-harness binary missing at {harness_bin}")
        return False
    if out_path.exists():
        out_path.unlink()
    cmd = [
        str(harness_bin),
        "scrape",
        "goodreads",
        "--profile",
        "book_show_api_chunked",
        "--set",
        f"book_id={book_id}",
        "--out",
        str(out_path),
    ]
    print(f"[session_health] probing book_id={book_id}")
    try:
        proc = subprocess.run(cmd, cwd=harness_root, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[session_health] probe failed: {exc}")
        return False
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-500:]
        print(f"[session_health] probe exited {proc.returncode}: {tail}")
        return False
    if not out_path.exists():
        return False
    try:
        record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    except (json.JSONDecodeError, IndexError, OSError):
        return False
    ok = bool(record.get("isbn13") and record.get("legacy_id") and not record.get("_scrape_warning"))
    print(f"[session_health] probe book_id={book_id} {'ok' if ok else 'failed'}")
    return ok


def _cooldown_seconds(tier: int, tiers: tuple[int, ...]) -> int:
    if not tiers:
        return DEFAULT_SPIKE_COOLDOWN_TIERS[-1]
    return tiers[min(tier, len(tiers) - 1)]


def post_rotate_cooldown(cooldown: int) -> int:
    """Idle the new exit briefly instead of the full 15m/45m/2h ladder."""
    return max(POST_ROTATE_COOLDOWN_FLOOR, min(cooldown, POST_ROTATE_COOLDOWN_CAP))


def _soft_block_escalate(
    *,
    harness_root: Path,
    jsonl_path: Path,
    catalog: Path,
    state_path: Path,
    max_tiers: int,
    cooldown_tiers: tuple[int, ...],
    sleep_fn: Callable[[float], None],
    skip_sleep: bool,
    wipe_fn: Callable[[Path], None],
    discover_fn: Callable[[Path], bool],
    probe_fn: Callable[[Path, int, Path], bool],
    rotate_fn: Callable[[], bool],
    state: RecoveryState,
) -> int:
    probe_book_id = pick_probe_book_id(jsonl_path, catalog)
    probe_out = catalog / "spike_probe.jsonl"
    while state.tier < max_tiers:
        cooldown = _cooldown_seconds(state.tier, cooldown_tiers)
        rotated = rotate_fn()
        if rotated:
            cooldown = post_rotate_cooldown(cooldown)
        print(
            f"[session_health] soft-block recovery tier {state.tier + 1}/{max_tiers} "
            f"(cooldown {cooldown}s, consecutive_spikes={state.consecutive_spikes}"
            f"{', rotated' if rotated else ''})"
        )
        wipe_fn(harness_root)
        discover_fn(harness_root)
        if not skip_sleep:
            print(f"[session_health] cooling down {cooldown}s before probe")
            sleep_fn(cooldown)
        if probe_fn(harness_root, probe_book_id, probe_out):
            reset_recovery_state(state_path)
            print("FORCE_BASELINE_PACING=1")
            return RECOVER_OK
        state.tier += 1
        save_recovery_state(state_path, state)
        print(f"[session_health] probe failed — escalating to tier {state.tier}")

    save_recovery_state(state_path, state)
    print(
        f"[session_health] unrecoverable soft-block after {max_tiers} tiers — "
        "change VPN exit, then re-run"
    )
    return RECOVER_HARD_STOP


def recover_from_spike(
    jsonl_path: Path,
    since_line: int,
    *,
    harness_root: Path,
    catalog_dir: Path | None = None,
    state_path: Path | None = None,
    max_tiers: int = DEFAULT_SPIKE_MAX_TIERS,
    cooldown_tiers: tuple[int, ...] = DEFAULT_SPIKE_COOLDOWN_TIERS,
    catalog_cooldown_seconds: int = DEFAULT_CATALOG_COOLDOWN_SECONDS,
    stale_retry_cooldowns: tuple[int, ...] = DEFAULT_STALE_BUILD_RETRY_COOLDOWNS,
    sleep_fn: Callable[[float], None] = time.sleep,
    wipe_fn: Callable[[Path], None] | None = None,
    discover_fn: Callable[[Path], bool] | None = None,
    probe_fn: Callable[[Path, int, Path], bool] | None = None,
    rotate_fn: Callable[[], bool] | None = None,
    skip_sleep: bool = False,
) -> int:
    """Classify the spike and recover. Returns RECOVER_OK / ESCALATED / HARD_STOP."""
    catalog = Path(catalog_dir) if catalog_dir is not None else Path(catalog_goodreads(""))
    state_path = state_path or (catalog / DEFAULT_STATE_NAME)
    stats = analyze_chunk(jsonl_path, since_line)
    kind = classify_chunk(jsonl_path, since_line, catalog_dir=catalog, stats=stats)
    print(f"[session_health] spike classified as {kind.value}")

    rotate = rotate_fn if rotate_fn is not None else maybe_rotate_exit

    if kind is SpikeKind.HARD_BLOCK:
        blocked = stats.warning_types.get(WARNING_BLOCKED, 0)
        if rotate():
            print(
                f"[session_health] hard block ({blocked}/{stats.warnings} warnings are "
                f"{WARNING_BLOCKED}) — rotated Mullvad exit, continuing"
            )
            print("FORCE_BASELINE_PACING=1")
            return RECOVER_OK
        print(
            f"[session_health] hard block ({blocked}/{stats.warnings} warnings are "
            f"{WARNING_BLOCKED}) — change VPN exit, then re-run"
        )
        return RECOVER_HARD_STOP

    if kind is SpikeKind.CATALOG:
        repaired, diagnosis = repair_catalog(catalog)
        if diagnosis.issues:
            print("[session_health] issues found:")
            for issue in diagnosis.issues:
                print(f"  - {issue}")
        if diagnosis.repairs:
            print("[session_health] repairs applied:")
            for repair in diagnosis.repairs:
                print(f"  - {repair}")
        if not skip_sleep:
            print(f"[session_health] catalog repaired — cooling down {catalog_cooldown_seconds}s")
            sleep_fn(catalog_cooldown_seconds)
        print("FORCE_BASELINE_PACING=1")
        return RECOVER_OK if repaired or diagnosis.issues else RECOVER_ESCALATED

    wipe = wipe_fn or wipe_chunked_profile
    discover = discover_fn or refresh_next_build
    probe = probe_fn or (lambda root, book_id, out: probe_known_good_book(root, book_id, out))
    probe_book_id = pick_probe_book_id(jsonl_path, catalog)
    probe_out = catalog / "spike_probe.jsonl"

    if kind is SpikeKind.STALE_BUILD:
        # Probe the same book discover just loaded (33), not the last JSONL
        # success — that id 404s on a dead build id even though it used to work.
        print("[session_health] stale Next.js build id — rediscovering immediately")
        probe_book_id = DEFAULT_PROBE_BOOK_ID
        discover(harness_root)
        if probe(harness_root, probe_book_id, probe_out):
            print("FORCE_BASELINE_PACING=1")
            return RECOVER_OK
        for i, cooldown in enumerate(stale_retry_cooldowns):
            print(
                f"[session_health] stale-build retry {i + 1}/{len(stale_retry_cooldowns)} "
                f"(cooldown {cooldown}s) — rediscover is cheap; 15m soft-block is not"
            )
            if not skip_sleep:
                sleep_fn(cooldown)
            discover(harness_root)
            if probe(harness_root, probe_book_id, probe_out):
                print("FORCE_BASELINE_PACING=1")
                return RECOVER_OK
        print("[session_health] probe still failing after rediscover retries — falling through to soft-block")

    state = load_recovery_state(state_path)
    state.consecutive_spikes += 1
    state.last_spike_at = _now_iso()
    return _soft_block_escalate(
        harness_root=harness_root,
        jsonl_path=jsonl_path,
        catalog=catalog,
        state_path=state_path,
        max_tiers=max_tiers,
        cooldown_tiers=cooldown_tiers,
        sleep_fn=sleep_fn,
        skip_sleep=skip_sleep,
        wipe_fn=wipe,
        discover_fn=discover,
        probe_fn=probe,
        rotate_fn=rotate,
        state=state,
    )


def cmd_check(args: argparse.Namespace) -> int:
    stats = analyze_chunk(Path(args.jsonl), args.since_line)
    print(
        f"[session_health] chunk since line {args.since_line}: "
        f"{stats.successes} ok / {stats.warnings} warn / {stats.total} total "
        f"({stats.warning_rate:.0%} warning rate, "
        f"{stats.trailing_warnings} trailing warns)"
    )
    if is_failure_spike(
        stats,
        min_records=args.min_chunk_records,
        threshold=args.failure_threshold,
        trailing_abort=args.trailing_abort,
    ):
        print(
            f"[session_health] FAILURE SPIKE: warning rate >= {args.failure_threshold:.0%} "
            f"with >= {args.min_chunk_records} records, or "
            f">= {args.trailing_abort} trailing warnings"
        )
        return 1
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    repaired, diagnosis = repair_catalog(Path(args.catalog_dir) if args.catalog_dir else None)
    if diagnosis.issues:
        print("[session_health] issues found:")
        for issue in diagnosis.issues:
            print(f"  - {issue}")
    if diagnosis.repairs:
        print("[session_health] repairs applied:")
        for repair in diagnosis.repairs:
            print(f"  - {repair}")
    if not diagnosis.issues:
        print("[session_health] no catalog issues detected")
        return 1
    if not repaired:
        print("[session_health] could not auto-repair — manual attention required")
        return 2
    top_line = ""
    ids_path = Path(args.catalog_dir) / "ids_remaining.txt" if args.catalog_dir else Path(catalog_goodreads("ids_remaining.txt"))
    if ids_path.exists():
        first = ids_path.read_text(encoding="utf-8").splitlines()[:1]
        if first:
            top_line = f" (top remaining book_id={first[0]})"
    print(f"[session_health] catalog repaired{top_line}")
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    max_tiers = args.max_tiers
    return recover_from_spike(
        Path(args.jsonl),
        args.since_line,
        harness_root=Path(args.harness_root),
        catalog_dir=Path(args.catalog_dir) if args.catalog_dir else None,
        max_tiers=max_tiers,
        skip_sleep=args.skip_sleep,
    )


def cmd_refresh_build(args: argparse.Namespace) -> int:
    ok = refresh_next_build(Path(args.harness_root))
    return 0 if ok else 1


def escalate_controlled_stop(
    *,
    state_path: Path,
    max_tiers: int = DEFAULT_SPIKE_MAX_TIERS,
    cooldown_tiers: tuple[int, ...] = DEFAULT_SPIKE_COOLDOWN_TIERS,
    sleep_fn: Callable[[float], None] = time.sleep,
    skip_sleep: bool = False,
) -> int:
    """Sleep the current soft-block tier after a harness exit 3. No VPN rotate."""
    state = load_recovery_state(state_path)
    state.consecutive_spikes += 1
    state.last_spike_at = _now_iso()
    if state.tier >= max_tiers:
        save_recovery_state(state_path, state)
        print(
            f"[session_health] unrecoverable after {max_tiers} controlled-stop tiers — "
            "stopping so a human can inspect"
        )
        return RECOVER_HARD_STOP
    cooldown = _cooldown_seconds(state.tier, cooldown_tiers)
    print(
        f"[session_health] controlled-stop cooldown tier {state.tier + 1}/{max_tiers} "
        f"({cooldown}s, consecutive_spikes={state.consecutive_spikes})"
    )
    if not skip_sleep:
        sleep_fn(cooldown)
    state.tier += 1
    save_recovery_state(state_path, state)
    return RECOVER_ESCALATED


def cmd_reset_state(args: argparse.Namespace) -> int:
    path = Path(args.state_path) if args.state_path else default_state_path(Path(args.catalog_dir) if args.catalog_dir else None)
    reset_recovery_state(path)
    print(f"[session_health] reset recovery state ({path})")
    return 0


def cmd_escalate_cooldown(args: argparse.Namespace) -> int:
    catalog = Path(args.catalog_dir) if args.catalog_dir else None
    state_path = Path(args.state_path) if args.state_path else default_state_path(catalog)
    return escalate_controlled_stop(
        state_path=state_path,
        max_tiers=args.max_tiers,
        skip_sleep=args.skip_sleep,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Analyze records written since --since-line")
    check.add_argument("--jsonl", type=Path, required=True)
    check.add_argument("--since-line", type=int, default=0, metavar="N")
    check.add_argument("--min-chunk-records", type=int, default=DEFAULT_MIN_CHUNK_RECORDS)
    check.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_SPIKE_THRESHOLD)
    check.add_argument("--trailing-abort", type=int, default=DEFAULT_TRAILING_WARN_ABORT)
    check.set_defaults(func=cmd_check)

    repair = sub.add_parser("repair", help="Diagnose and auto-fix catalog queue issues")
    repair.add_argument("--catalog-dir", type=Path, default=None)
    repair.set_defaults(func=cmd_repair)

    recover = sub.add_parser("recover", help="Classify a spike and run catalog/soft-block recovery")
    recover.add_argument("--jsonl", type=Path, required=True)
    recover.add_argument("--since-line", type=int, default=0, metavar="N")
    recover.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS_ROOT)
    recover.add_argument("--catalog-dir", type=Path, default=None)
    recover.add_argument(
        "--max-tiers",
        type=int,
        default=int(os.environ.get("SPIKE_MAX_TIERS", DEFAULT_SPIKE_MAX_TIERS)),
    )
    recover.add_argument(
        "--skip-sleep",
        action="store_true",
        help="Skip cooldowns (tests only).",
    )
    recover.set_defaults(func=cmd_recover)

    refresh = sub.add_parser(
        "refresh-build",
        help="Warmup home/list then rediscover the Next.js build id in the chunked Chrome profile",
    )
    refresh.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS_ROOT)
    refresh.set_defaults(func=cmd_refresh_build)

    reset = sub.add_parser("reset-state", help="Clear spike recovery escalation after a healthy chunk")
    reset.add_argument("--catalog-dir", type=Path, default=None)
    reset.add_argument("--state-path", type=Path, default=None)
    reset.set_defaults(func=cmd_reset_state)

    escalate = sub.add_parser(
        "escalate-cooldown",
        help="Sleep the current soft-block tier after a controlled scrape stop (exit 3). No VPN rotate.",
    )
    escalate.add_argument("--catalog-dir", type=Path, default=None)
    escalate.add_argument("--state-path", type=Path, default=None)
    escalate.add_argument(
        "--max-tiers",
        type=int,
        default=int(os.environ.get("SPIKE_MAX_TIERS", DEFAULT_SPIKE_MAX_TIERS)),
    )
    escalate.add_argument(
        "--skip-sleep",
        action="store_true",
        help="Skip cooldowns (tests only).",
    )
    escalate.set_defaults(func=cmd_escalate_cooldown)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
