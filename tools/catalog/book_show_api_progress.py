#!/usr/bin/env python3
"""Timestamped book_show_api scrape progress log + last-N-hours summary.

`book_show_api.jsonl` has no per-row timestamps. The orchestrator loop records
chunk boundaries here so "how many ISBN scrapes in the last 24h?" is a fast
lookup instead of an estimate.

Sidecars (under catalog_goodreads/):
  book_show_api_progress.jsonl  — one JSON object per chunk (append-only)
  book_show_api_chunk_open.json — written at chunk start; cleared on record

Usage:
    python tools/catalog/book_show_api_progress.py mark-start --since-line 24000
    python tools/catalog/book_show_api_progress.py record --since-line 24000
    python tools/catalog/book_show_api_progress.py --hours 24
    python tools/catalog/book_show_api_progress.py summarize --hours 24 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.book_show_api_session_health import (  # noqa: E402
    analyze_chunk,
    is_failure_spike,
)
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_PROGRESS_NAME = "book_show_api_progress.jsonl"
DEFAULT_OPEN_NAME = "book_show_api_chunk_open.json"
DEFAULT_JSONL_NAME = "book_show_api.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def default_progress_path(catalog_dir: Path | None = None) -> Path:
    if catalog_dir is not None:
        return Path(catalog_dir) / DEFAULT_PROGRESS_NAME
    return catalog_goodreads(DEFAULT_PROGRESS_NAME)


def default_open_path(catalog_dir: Path | None = None) -> Path:
    if catalog_dir is not None:
        return Path(catalog_dir) / DEFAULT_OPEN_NAME
    return catalog_goodreads(DEFAULT_OPEN_NAME)


def default_jsonl_path(catalog_dir: Path | None = None) -> Path:
    if catalog_dir is not None:
        return Path(catalog_dir) / DEFAULT_JSONL_NAME
    return catalog_goodreads(DEFAULT_JSONL_NAME)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def mark_chunk_start(
    *,
    since_line: int,
    jsonl_path: Path,
    open_path: Path,
    ts: str | None = None,
) -> dict:
    """Persist the open-chunk baseline so mid-flight scrapes are countable."""
    record = {
        "ts": ts or _now_iso(),
        "event": "chunk_start",
        "jsonl": str(jsonl_path),
        "since_line": int(since_line),
        "lines_after": int(since_line),
        "attempts": 0,
        "isbn_ok": 0,
        "warnings": 0,
    }
    open_path.parent.mkdir(parents=True, exist_ok=True)
    open_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def record_chunk(
    *,
    since_line: int,
    jsonl_path: Path,
    progress_path: Path,
    open_path: Path | None = None,
    event: str | None = None,
    ts: str | None = None,
) -> dict:
    """Append chunk stats to the progress log and clear any open-chunk marker."""
    stats = analyze_chunk(jsonl_path, since_line=since_line)
    lines_after = count_lines(jsonl_path)
    if event is None:
        event = (
            "chunk_spike"
            if is_failure_spike(stats)
            else "chunk"
        )
    record = {
        "ts": ts or _now_iso(),
        "event": event,
        "jsonl": str(jsonl_path),
        "since_line": int(since_line),
        "lines_after": lines_after,
        "attempts": stats.total,
        "isbn_ok": stats.successes,
        "warnings": stats.warnings,
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    if open_path is not None and open_path.exists():
        open_path.unlink()
    return record


def load_progress(progress_path: Path) -> list[dict]:
    if not progress_path.exists():
        return []
    out: list[dict] = []
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def load_open_chunk(open_path: Path) -> dict | None:
    if not open_path.exists():
        return None
    try:
        data = json.loads(open_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class ProgressSummary:
    hours: float
    attempts: int
    isbn_ok: int
    warnings: int
    closed_chunks: int
    open_attempts: int
    open_isbn_ok: int
    open_warnings: int
    progress_path: Path
    since: datetime
    until: datetime
    has_progress_log: bool

    @property
    def total_attempts(self) -> int:
        return self.attempts + self.open_attempts

    @property
    def total_isbn_ok(self) -> int:
        return self.isbn_ok + self.open_isbn_ok

    @property
    def total_warnings(self) -> int:
        return self.warnings + self.open_warnings

    @property
    def isbn_per_hour(self) -> float:
        if self.hours <= 0:
            return 0.0
        return self.total_isbn_ok / self.hours

    def to_dict(self) -> dict:
        return {
            "hours": self.hours,
            "since": self.since.isoformat(timespec="seconds"),
            "until": self.until.isoformat(timespec="seconds"),
            "attempts": self.total_attempts,
            "isbn_ok": self.total_isbn_ok,
            "isbn_per_hour": self.isbn_per_hour,
            "warnings": self.total_warnings,
            "closed": {
                "chunks": self.closed_chunks,
                "attempts": self.attempts,
                "isbn_ok": self.isbn_ok,
                "warnings": self.warnings,
            },
            "open": {
                "attempts": self.open_attempts,
                "isbn_ok": self.open_isbn_ok,
                "warnings": self.open_warnings,
            },
            "progress_path": str(self.progress_path),
            "has_progress_log": self.has_progress_log,
        }


def summarize(
    *,
    hours: float = 24.0,
    progress_path: Path,
    open_path: Path,
    jsonl_path: Path,
    now: datetime | None = None,
) -> ProgressSummary:
    until = now or datetime.now(timezone.utc)
    since = until - timedelta(hours=hours)
    rows = load_progress(progress_path)
    attempts = isbn_ok = warnings = closed = 0
    for row in rows:
        ts_raw = row.get("ts")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = _parse_ts(ts_raw)
        except ValueError:
            continue
        if ts < since or ts > until:
            continue
        event = row.get("event")
        if event not in {"chunk", "chunk_spike"}:
            continue
        closed += 1
        attempts += int(row.get("attempts") or 0)
        isbn_ok += int(row.get("isbn_ok") or 0)
        warnings += int(row.get("warnings") or 0)

    open_attempts = open_isbn_ok = open_warnings = 0
    open_rec = load_open_chunk(open_path)
    if open_rec is not None:
        include_open = True
        ts_raw = open_rec.get("ts")
        if isinstance(ts_raw, str):
            try:
                open_ts = _parse_ts(ts_raw)
                # Drop stale open markers from before the window (e.g. loop
                # crashed without record); in-window open chunks stay.
                if open_ts < since:
                    include_open = False
            except ValueError:
                pass
        if include_open:
            since_line = int(open_rec.get("since_line") or 0)
            stats = analyze_chunk(jsonl_path, since_line=since_line)
            open_attempts = stats.total
            open_isbn_ok = stats.successes
            open_warnings = stats.warnings

    return ProgressSummary(
        hours=hours,
        attempts=attempts,
        isbn_ok=isbn_ok,
        warnings=warnings,
        closed_chunks=closed,
        open_attempts=open_attempts,
        open_isbn_ok=open_isbn_ok,
        open_warnings=open_warnings,
        progress_path=progress_path,
        since=since,
        until=until,
        has_progress_log=progress_path.exists() and bool(rows),
    )


def format_summary(summary: ProgressSummary) -> str:
    lines = [
        f"book_show_api scrapes — last {summary.hours:g}h "
        f"({summary.since.astimezone().strftime('%Y-%m-%d %H:%M')} → "
        f"{summary.until.astimezone().strftime('%Y-%m-%d %H:%M')} local)",
        f"  attempts:  {summary.total_attempts}",
        f"  isbn_ok:   {summary.total_isbn_ok}",
        f"  isbn/hour: {summary.isbn_per_hour:.1f}",
        f"  warnings:  {summary.total_warnings}",
    ]
    if summary.open_attempts:
        lines.append(
            f"  (includes open chunk: {summary.open_attempts} attempts, "
            f"{summary.open_isbn_ok} isbn_ok, {summary.open_warnings} warnings)"
        )
    if summary.closed_chunks:
        lines.append(f"  closed chunks in window: {summary.closed_chunks}")
    if not summary.has_progress_log and not summary.open_attempts:
        lines.append(
            "  note: no progress log yet — counts start after the loop records "
            "chunks via mark-start/record (see docs/BOOK_CATALOG.md)"
        )
    elif not summary.has_progress_log and summary.open_attempts:
        lines.append(
            "  note: progress log empty; showing open chunk only "
            f"(from {summary.progress_path.name} sibling open marker)"
        )
    return "\n".join(lines)


def format_hour_rate(summary: ProgressSummary) -> str:
    return (
        f"last {summary.hours:g}h: {summary.total_isbn_ok} isbn_ok "
        f"({summary.isbn_per_hour:.1f}/h)"
    )


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="Override catalog_goodreads() directory",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=f"Default: catalog_goodreads({DEFAULT_JSONL_NAME!r})",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help=f"Default: catalog_goodreads({DEFAULT_PROGRESS_NAME!r})",
    )
    parser.add_argument(
        "--open",
        dest="open_path",
        type=Path,
        default=None,
        help=f"Default: catalog_goodreads({DEFAULT_OPEN_NAME!r})",
    )


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    catalog = args.catalog_dir
    jsonl = args.jsonl or default_jsonl_path(catalog)
    progress = args.progress or default_progress_path(catalog)
    open_path = args.open_path or default_open_path(catalog)
    return jsonl, progress, open_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record / summarize book_show_api scrape progress by time window.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_start = sub.add_parser("mark-start", help="Record open-chunk baseline before a scrape")
    _add_shared_paths(p_start)
    p_start.add_argument("--since-line", type=int, required=True)

    p_rec = sub.add_parser("record", help="Append closed-chunk stats after a scrape")
    _add_shared_paths(p_rec)
    p_rec.add_argument("--since-line", type=int, required=True)
    p_rec.add_argument(
        "--event",
        choices=("chunk", "chunk_spike"),
        default=None,
        help="Override auto classification (default: spike detector)",
    )

    p_sum = sub.add_parser("summarize", help="Print totals for the last N hours")
    _add_shared_paths(p_sum)
    p_sum.add_argument("--hours", type=float, default=24.0)
    p_sum.add_argument("--json", action="store_true")

    # Default command: summarize --hours 24
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--json", action="store_true", default=False)
    _add_shared_paths(parser)

    args = parser.parse_args(argv)
    cmd = args.cmd
    if cmd is None:
        cmd = "summarize"
        if args.hours is None:
            args.hours = 24.0

    jsonl, progress, open_path = _resolve_paths(args)

    if cmd == "mark-start":
        rec = mark_chunk_start(
            since_line=args.since_line,
            jsonl_path=jsonl,
            open_path=open_path,
        )
        print(
            f"[book_show_api_progress] chunk_start since_line={rec['since_line']} "
            f"-> {open_path}"
        )
        return 0

    if cmd == "record":
        rec = record_chunk(
            since_line=args.since_line,
            jsonl_path=jsonl,
            progress_path=progress,
            open_path=open_path,
            event=args.event,
        )
        hour = summarize(
            hours=1,
            progress_path=progress,
            open_path=open_path,
            jsonl_path=jsonl,
        )
        print(
            f"[book_show_api_progress] {rec['event']}: "
            f"{rec['isbn_ok']} isbn_ok / {rec['warnings']} warn / "
            f"{rec['attempts']} total (lines {rec['since_line']}→{rec['lines_after']}) "
            f"| {format_hour_rate(hour)}"
        )
        return 0

    if cmd == "summarize":
        hours = float(args.hours if args.hours is not None else 24.0)
        summary = summarize(
            hours=hours,
            progress_path=progress,
            open_path=open_path,
            jsonl_path=jsonl,
        )
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(format_summary(summary))
        return 0

    parser.error(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
