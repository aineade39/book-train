#!/usr/bin/env python3
"""Unit tests for tools/scrape_goodreads_lists.py (run: python tools/test_scrape_goodreads_lists.py).

No network access and no real scrape-harness invocation: subprocess.run is
mocked throughout, exercising the checkpoint/retry/session-cap/challenge
logic in isolation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.scrape_goodreads_lists import (  # noqa: E402
    STATUS_CHALLENGED,
    STATUS_DONE,
    STATUS_EMPTY_UNEXPECTED,
    STATUS_ERROR,
    STATUS_PENDING,
    Checkpoint,
    SeedList,
    _count_books,
    _looks_like_challenge,
    load_seed_lists,
    main,
    run_one_list,
)

SEED_YAML = """
lists:
  - list_id: 1
    slug: Alpha
    genre: general
  - list_id: 2
    slug: Beta
    genre: fantasy
"""


class TestLoadSeedLists(unittest.TestCase):
    def test_parses_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed.yaml"
            path.write_text(SEED_YAML, encoding="utf-8")
            seeds = load_seed_lists(path)
        self.assertEqual(seeds, [SeedList(1, "Alpha", "general"), SeedList(2, "Beta", "fantasy")])


class TestCheckpoint(unittest.TestCase):
    def test_upsert_then_pending_lists(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general"), SeedList(2, "Beta", "fantasy")])
            pending = cp.pending_lists(max_attempts=3)
            self.assertEqual({r["list_id"] for r in pending}, {1, 2})
            self.assertTrue(all(r["status"] == STATUS_PENDING for r in pending))

    def test_record_result_marks_done_and_excludes_from_pending(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path="/tmp/x.jsonl", book_count=42)
            self.assertEqual(cp.pending_lists(max_attempts=3), [])
            row = cp.rows()[0]
            self.assertEqual(row["status"], STATUS_DONE)
            self.assertEqual(row["book_count"], 42)
            self.assertEqual(row["attempts"], 1)

    def test_error_stays_pending_until_max_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            self.assertEqual(len(cp.pending_lists(max_attempts=3)), 1)
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            self.assertEqual(len(cp.pending_lists(max_attempts=3)), 1)
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            self.assertEqual(len(cp.pending_lists(max_attempts=3)), 0, "3rd attempt should exhaust max_attempts=3")

    def test_upsert_does_not_reset_status_of_existing_list(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path="/tmp/x.jsonl", book_count=10)
            cp.upsert_seed([SeedList(1, "Alpha-Renamed", "general")])
            row = cp.rows()[0]
            self.assertEqual(row["status"], STATUS_DONE, "re-seeding must not un-do a completed list")
            self.assertEqual(row["slug"], "Alpha-Renamed")

    def test_reset_list_clears_done_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path="/tmp/x.jsonl", book_count=10000)
            self.assertTrue(cp.reset_list(1))
            row = cp.rows()[0]
            self.assertEqual(row["status"], STATUS_PENDING)
            self.assertEqual(row["attempts"], 0)
            self.assertIsNone(row["last_error"])
            self.assertIsNone(row["book_count"])
            self.assertEqual(len(cp.pending_lists(max_attempts=3)), 1)

    def test_reset_list_returns_false_for_unknown_id(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            self.assertFalse(cp.reset_list(999))

    def test_summary_counts_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "A", "g"), SeedList(2, "B", "g")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path="x", book_count=5)
            summary = cp.summary()
        self.assertEqual(summary, {STATUS_DONE: 1, STATUS_PENDING: 1})


class TestCountBooks(unittest.TestCase):
    def test_sums_book_urls_across_page_records(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jsonl"
            path.write_text(
                json.dumps({"book_urls": ["/a", "/b"]}) + "\n" + json.dumps({"book_urls": ["/c"]}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_count_books(path), 3)

    def test_missing_file_is_zero(self) -> None:
        self.assertEqual(_count_books(Path("/nonexistent/out.jsonl")), 0)

    def test_empty_file_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jsonl"
            path.write_text("", encoding="utf-8")
            self.assertEqual(_count_books(path), 0)


class TestLooksLikeChallenge(unittest.TestCase):
    def test_matches_known_markers_case_insensitively(self) -> None:
        self.assertTrue(_looks_like_challenge("Please complete this CAPTCHA to continue"))
        self.assertTrue(_looks_like_challenge("Pardon Our Interruption..."))

    def test_plain_error_does_not_match(self) -> None:
        self.assertFalse(_looks_like_challenge("TimeoutError: waiting for selector 'table.tableList' failed"))


class TestRunOneList(unittest.TestCase):
    def test_success_with_books_is_done(self) -> None:
        seed = SeedList(367, "Best_Fantasy_Books", "fantasy")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)

            def fake_run(cmd, cwd, capture_output, text, timeout):
                out_path = Path(cmd[cmd.index("--out") + 1])
                out_path.write_text(json.dumps({"book_urls": ["/a", "/b"]}) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                status, error, out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"), Path("/fake/root"), seed, out_dir, skip_vpn_check=False
                )
        self.assertEqual(status, STATUS_DONE)
        self.assertIsNone(error)
        self.assertEqual(book_count, 2)

    def test_success_with_zero_books_is_empty_unexpected(self) -> None:
        seed = SeedList(1, "Empty", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)

            def fake_run(cmd, cwd, capture_output, text, timeout):
                out_path = Path(cmd[cmd.index("--out") + 1])
                out_path.write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"), Path("/fake/root"), seed, out_dir, skip_vpn_check=False
                )
        self.assertEqual(status, STATUS_EMPTY_UNEXPECTED)
        self.assertEqual(book_count, 0)
        self.assertIsNotNone(error)

    def test_nonzero_exit_with_challenge_marker_is_challenged(self) -> None:
        seed = SeedList(1, "X", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            proc = subprocess.CompletedProcess([], 1, stdout="", stderr="Pardon Our Interruption")
            with patch("tools.scrape_goodreads_lists.subprocess.run", return_value=proc):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"), Path("/fake/root"), seed, out_dir, skip_vpn_check=False
                )
        self.assertEqual(status, STATUS_CHALLENGED)
        self.assertIn("Pardon", error or "")
        self.assertIsNone(book_count)

    def test_nonzero_exit_without_challenge_marker_is_error(self) -> None:
        seed = SeedList(1, "X", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            proc = subprocess.CompletedProcess([], 1, stdout="", stderr="TimeoutError: net::ERR_CONNECTION_RESET")
            with patch("tools.scrape_goodreads_lists.subprocess.run", return_value=proc):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"), Path("/fake/root"), seed, out_dir, skip_vpn_check=False
                )
        self.assertEqual(status, STATUS_ERROR)
        self.assertIn("ERR_CONNECTION_RESET", error or "")

    def test_timeout_is_error(self) -> None:
        seed = SeedList(1, "X", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            with patch(
                "tools.scrape_goodreads_lists.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1),
            ):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"), Path("/fake/root"), seed, out_dir, skip_vpn_check=False
                )
        self.assertEqual(status, STATUS_ERROR)
        self.assertIn("timed out", error or "")


class TestMainEndToEnd(unittest.TestCase):
    """Drives main() with a fake harness binary + mocked subprocess to check
    restartability: a second invocation must skip lists already `done`."""

    def _run_main(self, tmp: Path, seed_yaml: str, extra_args: list[str], fake_run) -> int:
        seed_path = tmp / "seed.yaml"
        seed_path.write_text(seed_yaml, encoding="utf-8")
        harness_root = tmp / "harness"
        (harness_root / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        harness_bin = harness_root / ".venv" / "bin" / "scrape-harness"
        harness_bin.write_text("#!/bin/sh\n")
        harness_bin.chmod(0o755)

        args = [
            "--seed-lists",
            str(seed_path),
            "--harness-root",
            str(harness_root),
            "--out-dir",
            str(tmp / "raw"),
            "--checkpoint-db",
            str(tmp / "checkpoint.sqlite"),
            "--shuffle-seed",
            "1",
            *extra_args,
        ]
        with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
            with patch("tools.scrape_goodreads_lists.time.sleep"):
                return main(args)

    def test_second_run_skips_already_done_lists(self) -> None:
        call_count = {"n": 0}

        def fake_run(cmd, cwd, capture_output, text, timeout):
            call_count["n"] += 1
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc1 = self._run_main(tmp, SEED_YAML, ["--session-max-lists", "10"], fake_run)
            self.assertEqual(rc1, 0)
            self.assertEqual(call_count["n"], 2, "first run should scrape both seed lists")

            rc2 = self._run_main(tmp, SEED_YAML, ["--session-max-lists", "10"], fake_run)
            self.assertEqual(rc2, 0)
            self.assertEqual(call_count["n"], 2, "second run should find nothing pending and not re-invoke harness")

    def test_session_max_lists_caps_a_single_run(self) -> None:
        call_count = {"n": 0}

        def fake_run(cmd, cwd, capture_output, text, timeout):
            call_count["n"] += 1
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc = self._run_main(tmp, SEED_YAML, ["--session-max-lists", "1"], fake_run)
            self.assertEqual(rc, 0)
            self.assertEqual(call_count["n"], 1, "session cap of 1 must stop after the first list")

    def test_challenge_stops_session_early(self) -> None:
        call_count = {"n": 0}

        def fake_run(cmd, cwd, capture_output, text, timeout):
            call_count["n"] += 1
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Pardon Our Interruption")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc = self._run_main(tmp, SEED_YAML, ["--session-max-lists", "10"], fake_run)
            self.assertEqual(rc, 0)
            self.assertEqual(call_count["n"], 1, "a challenge on the first list must stop the whole run")

    def test_report_only_does_not_invoke_harness(self) -> None:
        def fake_run(*_a, **_k):
            raise AssertionError("subprocess.run should never be called with --report-only")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc = self._run_main(tmp, SEED_YAML, ["--report-only"], fake_run)
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
