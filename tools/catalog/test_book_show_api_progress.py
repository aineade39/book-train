#!/usr/bin/env python3
"""Unit tests for tools/catalog/book_show_api_progress.py
(run: python tools/catalog/test_book_show_api_progress.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.book_show_api_progress import (  # noqa: E402
    format_hour_rate,
    format_summary,
    main,
    mark_chunk_start,
    record_chunk,
    summarize,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestBookShowApiProgress(unittest.TestCase):
    def test_mark_start_and_record_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "book_show_api.jsonl"
            progress = catalog / "book_show_api_progress.jsonl"
            open_path = catalog / "book_show_api_chunk_open.json"
            _write_jsonl(
                jsonl,
                [
                    {"legacy_id": 1, "isbn13": "9780000000001"},
                    {"legacy_id": 2, "isbn13": "9780000000002"},
                ],
            )
            mark_chunk_start(
                since_line=2,
                jsonl_path=jsonl,
                open_path=open_path,
                ts="2026-08-20T15:00:00+00:00",
            )
            self.assertTrue(open_path.exists())

            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"legacy_id": 3, "isbn13": "9780000000003"}) + "\n")
                f.write(json.dumps({"_scrape_warning": "incomplete_record"}) + "\n")

            rec = record_chunk(
                since_line=2,
                jsonl_path=jsonl,
                progress_path=progress,
                open_path=open_path,
                ts="2026-08-20T15:10:00+00:00",
            )
            self.assertEqual(rec["event"], "chunk")
            self.assertEqual(rec["attempts"], 2)
            self.assertEqual(rec["isbn_ok"], 1)
            self.assertEqual(rec["warnings"], 1)
            self.assertFalse(open_path.exists())
            rows = [json.loads(line) for line in progress.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["isbn_ok"], 1)

    def test_summarize_window_and_open_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "book_show_api.jsonl"
            progress = catalog / "book_show_api_progress.jsonl"
            open_path = catalog / "book_show_api_chunk_open.json"
            now = datetime(2026, 8, 20, 16, 0, tzinfo=timezone.utc)
            old = (now - timedelta(hours=30)).isoformat(timespec="seconds")
            recent = (now - timedelta(hours=2)).isoformat(timespec="seconds")
            with progress.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": old,
                            "event": "chunk",
                            "attempts": 100,
                            "isbn_ok": 90,
                            "warnings": 10,
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "ts": recent,
                            "event": "chunk",
                            "attempts": 50,
                            "isbn_ok": 40,
                            "warnings": 10,
                        }
                    )
                    + "\n"
                )
            _write_jsonl(
                jsonl,
                [
                    {"legacy_id": 1, "isbn13": "9780000000001"},
                    {"legacy_id": 2, "isbn13": "9780000000002"},
                    {"_scrape_warning": "incomplete_record"},
                ],
            )
            mark_chunk_start(
                since_line=1,
                jsonl_path=jsonl,
                open_path=open_path,
                ts=now.isoformat(timespec="seconds"),
            )
            summary = summarize(
                hours=24,
                progress_path=progress,
                open_path=open_path,
                jsonl_path=jsonl,
                now=now,
            )
            self.assertEqual(summary.attempts, 50)
            self.assertEqual(summary.isbn_ok, 40)
            self.assertEqual(summary.closed_chunks, 1)
            self.assertEqual(summary.open_attempts, 2)
            self.assertEqual(summary.open_isbn_ok, 1)
            self.assertEqual(summary.open_warnings, 1)
            self.assertEqual(summary.total_isbn_ok, 41)
            text = format_summary(summary)
            self.assertIn("isbn_ok:   41", text)
            self.assertIn("isbn/hour: 1.7", text)
            self.assertIn("open chunk", text)
            self.assertAlmostEqual(summary.isbn_per_hour, 41 / 24)

    def test_one_hour_rate_is_isbn_ok_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "book_show_api.jsonl"
            progress = catalog / "book_show_api_progress.jsonl"
            open_path = catalog / "book_show_api_chunk_open.json"
            now = datetime(2026, 8, 20, 16, 0, tzinfo=timezone.utc)
            recent = (now - timedelta(minutes=20)).isoformat(timespec="seconds")
            with progress.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": recent,
                            "event": "chunk",
                            "attempts": 12,
                            "isbn_ok": 12,
                            "warnings": 0,
                        }
                    )
                    + "\n"
                )
            jsonl.write_text("", encoding="utf-8")
            summary = summarize(
                hours=1,
                progress_path=progress,
                open_path=open_path,
                jsonl_path=jsonl,
                now=now,
            )
            self.assertEqual(summary.total_isbn_ok, 12)
            self.assertAlmostEqual(summary.isbn_per_hour, 12.0)
            self.assertIn("12.0/h", format_hour_rate(summary))
            text = format_summary(summary)
            self.assertIn("isbn/hour: 12.0", text)

    def test_record_cli_includes_last_1h(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "book_show_api.jsonl"
            _write_jsonl(jsonl, [{"legacy_id": 1, "isbn13": "9780000000001"}])
            from io import StringIO
            from contextlib import redirect_stdout

            buf = StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "record",
                        "--catalog-dir",
                        str(catalog),
                        "--since-line",
                        "0",
                    ]
                )
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("last 1h", out)
            self.assertIn("/h)", out)

    def test_summarize_ignores_stale_open_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "book_show_api.jsonl"
            progress = catalog / "book_show_api_progress.jsonl"
            open_path = catalog / "book_show_api_chunk_open.json"
            progress.write_text("", encoding="utf-8")
            now = datetime(2026, 8, 20, 16, 0, tzinfo=timezone.utc)
            _write_jsonl(
                jsonl,
                [
                    {"legacy_id": 1, "isbn13": "9780000000001"},
                    {"_scrape_warning": "incomplete_record"},
                ],
            )
            mark_chunk_start(
                since_line=0,
                jsonl_path=jsonl,
                open_path=open_path,
                ts=(now - timedelta(hours=30)).isoformat(timespec="seconds"),
            )
            summary = summarize(
                hours=24,
                progress_path=progress,
                open_path=open_path,
                jsonl_path=jsonl,
                now=now,
            )
            self.assertEqual(summary.open_attempts, 0)
            self.assertEqual(summary.total_attempts, 0)

    def test_cli_summarize_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            (catalog / "book_show_api.jsonl").write_text("", encoding="utf-8")
            rc = main(["summarize", "--catalog-dir", str(catalog), "--hours", "24", "--json"])
            self.assertEqual(rc, 0)

    def test_cli_default_is_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            (catalog / "book_show_api.jsonl").write_text("", encoding="utf-8")
            rc = main(["--catalog-dir", str(catalog), "--hours", "12"])
            self.assertEqual(rc, 0)

    def test_record_marks_spike_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "book_show_api.jsonl"
            progress = catalog / "book_show_api_progress.jsonl"
            records = [{"_scrape_warning": "incomplete_record"} for _ in range(25)]
            _write_jsonl(jsonl, records)
            rec = record_chunk(
                since_line=0,
                jsonl_path=jsonl,
                progress_path=progress,
                open_path=catalog / "open.json",
                ts="2026-08-20T15:00:00+00:00",
            )
            self.assertEqual(rec["event"], "chunk_spike")
            self.assertEqual(rec["isbn_ok"], 0)
            self.assertEqual(rec["warnings"], 25)


if __name__ == "__main__":
    unittest.main()
