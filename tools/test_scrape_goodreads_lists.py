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
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.scrape_goodreads_lists import (  # noqa: E402
    CHUNKED_PROFILE_ID,
    RETRY_BACKOFF_MINUTES,
    SOFT_BLOCK_THRESHOLD,
    STATUS_CHALLENGED,
    STATUS_CHUNKED,
    STATUS_DONE,
    STATUS_EMPTY_UNEXPECTED,
    STATUS_ERROR,
    STATUS_PENDING,
    Checkpoint,
    SeedList,
    _count_books,
    _count_saved_pages,
    _looks_like_challenge,
    _now_iso,
    _write_chunked_profile,
    load_deprecated_list_ids,
    load_seed_lists,
    main,
    run_one_list,
)

FAKE_LIST_SHOW_PROFILE = """\
id: list_show
start_url: "https://example.com/list/show/{list_id}.{slug}"
steps: []
pagination:
  type: url_pattern
  url_pattern: "https://example.com/list/show/{list_id}.{slug}?page={page}"
  start_page: 1
  max_pages: 1000
policy:
  delay_ms_min: 1
  delay_ms_max: 1
  max_requests: 50
"""


def _write_fake_harness(harness_root: Path) -> Path:
    """Builds a fake scrape-harness checkout: console script + the
    `sites/goodreads/profiles/list_show.yaml` that `_write_chunked_profile`
    (called from `main()`) reads as its base profile."""
    (harness_root / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    harness_bin = harness_root / ".venv" / "bin" / "scrape-harness"
    harness_bin.write_text("#!/bin/sh\n")
    harness_bin.chmod(0o755)
    profiles_dir = harness_root / "sites" / "goodreads" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "list_show.yaml").write_text(FAKE_LIST_SHOW_PROFILE, encoding="utf-8")
    return harness_bin

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


class TestLoadDeprecatedListIds(unittest.TestCase):
    def test_missing_file_returns_empty_set(self) -> None:
        self.assertEqual(load_deprecated_list_ids(Path("/nonexistent/overrides.yaml")), set())

    def test_empty_overrides_file_returns_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "overrides.yaml"
            path.write_text("overrides: {}\n", encoding="utf-8")
            self.assertEqual(load_deprecated_list_ids(path), set())

    def test_populated_file_returns_only_deprecated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "overrides.yaml"
            path.write_text(
                "overrides:\n"
                "  3:\n"
                "    curation_status: deprecated\n"
                "    reason: subset overlap\n"
                "  15:\n"
                "    curation_status: active\n",
                encoding="utf-8",
            )
            self.assertEqual(load_deprecated_list_ids(path), {3})


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
            # `attempts` tracks *consecutive non-progress* attempts (see
            # record_result) and resets to 0 on success — a finished list
            # isn't "1 attempt away" from max_attempts.
            self.assertEqual(row["attempts"], 0)

    def test_error_stays_pending_until_max_attempts(self) -> None:
        """Bypasses the retry-backoff cooldown between calls (set next_retry_at
        to now) so this test isolates max_attempts exhaustion; cooldown timing
        itself is covered by test_error_sets_increasing_backoff_and_excludes_from_pending_until_elapsed."""
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            cp.conn.execute("UPDATE goodreads_lists SET next_retry_at = ? WHERE list_id = 1", (_now_iso(),))
            self.assertEqual(len(cp.pending_lists(max_attempts=3)), 1)
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            cp.conn.execute("UPDATE goodreads_lists SET next_retry_at = ? WHERE list_id = 1", (_now_iso(),))
            self.assertEqual(len(cp.pending_lists(max_attempts=3)), 1)
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            cp.conn.execute("UPDATE goodreads_lists SET next_retry_at = ? WHERE list_id = 1", (_now_iso(),))
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

    def test_chunked_is_scheduled_like_pending_without_consuming_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            row = cp.record_result(1, status=STATUS_CHUNKED, error=None, output_path="/tmp/x.jsonl", book_count=50)
            self.assertEqual(row["status"], STATUS_CHUNKED)
            self.assertEqual(row["attempts"], 0, "a healthy chunk boundary is forward progress, not a failure")
            self.assertIsNone(row["next_retry_at"])
            pending = cp.pending_lists(max_attempts=3)
            self.assertEqual({r["list_id"] for r in pending}, {1})

    def test_done_after_errors_resets_attempts_and_timeout_streak(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            row = cp.record_result(1, status=STATUS_DONE, error=None, output_path="/tmp/x.jsonl", book_count=100)
            self.assertEqual(row["attempts"], 0)
            self.assertEqual(row["consecutive_plain_timeouts"], 0)
            self.assertIsNone(row["next_retry_at"])

    def test_error_sets_increasing_backoff_and_excludes_from_pending_until_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])

            row1 = cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            self.assertEqual(cp.pending_lists(max_attempts=3), [], "fresh error must cool down before retry")
            delay1 = datetime.fromisoformat(row1["next_retry_at"]) - datetime.now(timezone.utc)
            self.assertGreater(delay1.total_seconds(), (RETRY_BACKOFF_MINUTES[0] - 1) * 60)

            # Force the cooldown to have already elapsed, then error again —
            # the second consecutive error should use a *longer* cooldown tier.
            cp.conn.execute("UPDATE goodreads_lists SET next_retry_at = ? WHERE list_id = 1", (_now_iso(),))
            cp.conn.commit()
            self.assertEqual(len(cp.pending_lists(max_attempts=5)), 1)
            row2 = cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            delay2 = datetime.fromisoformat(row2["next_retry_at"]) - datetime.now(timezone.utc)
            self.assertGreater(delay2.total_seconds(), (RETRY_BACKOFF_MINUTES[1] - 1) * 60)
            self.assertGreater(delay2.total_seconds(), delay1.total_seconds())

    def test_consecutive_plain_timeouts_increments_on_error_and_resets_on_progress(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            row = cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            self.assertEqual(row["consecutive_plain_timeouts"], 1)
            cp.conn.execute("UPDATE goodreads_lists SET next_retry_at = ? WHERE list_id = 1", (_now_iso(),))
            row = cp.record_result(1, status=STATUS_ERROR, error="boom", output_path=None, book_count=None)
            self.assertEqual(row["consecutive_plain_timeouts"], 2)
            row = cp.record_result(1, status=STATUS_CHUNKED, error=None, output_path="/x", book_count=10)
            self.assertEqual(row["consecutive_plain_timeouts"], 0, "forward progress clears the timeout streak")

    def test_challenged_does_not_set_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            row = cp.record_result(1, status=STATUS_CHALLENGED, error="captcha", output_path=None, book_count=None)
            self.assertIsNone(row["next_retry_at"], "challenged requires manual --redo-list, not an auto cooldown")

    def test_summary_counts_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpoint(Path(d) / "cp.sqlite")
            cp.upsert_seed([SeedList(1, "A", "g"), SeedList(2, "B", "g")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path="x", book_count=5)
            summary = cp.summary()
        self.assertEqual(summary, {STATUS_DONE: 1, STATUS_PENDING: 1})


class TestCountBooks(unittest.TestCase):
    def test_count_saved_pages_counts_non_empty_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jsonl"
            path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n\n" + json.dumps({"book_urls": ["/b"]}) + "\n")
            self.assertEqual(_count_saved_pages(path), 2)

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


class TestWriteChunkedProfile(unittest.TestCase):
    def test_writes_profile_with_overridden_max_requests(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as d:
            harness_root = Path(d) / "harness"
            _write_fake_harness(harness_root)
            profile_id = _write_chunked_profile(harness_root, 7)
            self.assertEqual(profile_id, CHUNKED_PROFILE_ID)
            out_path = harness_root / "sites" / "goodreads" / "profiles" / f"{CHUNKED_PROFILE_ID}.yaml"
            self.assertTrue(out_path.exists())
            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["id"], CHUNKED_PROFILE_ID)
            self.assertEqual(data["policy"]["max_requests"], 7)
            # Everything else should still mirror the base profile.
            self.assertEqual(data["pagination"]["type"], "url_pattern")

    def test_missing_base_profile_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            harness_root = Path(d) / "harness"
            with self.assertRaises(FileNotFoundError):
                _write_chunked_profile(harness_root, 50)

    def test_rewritten_on_every_call(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            harness_root = Path(d) / "harness"
            _write_fake_harness(harness_root)
            _write_chunked_profile(harness_root, 10)
            _write_chunked_profile(harness_root, 20)
            import yaml

            out_path = harness_root / "sites" / "goodreads" / "profiles" / f"{CHUNKED_PROFILE_ID}.yaml"
            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["policy"]["max_requests"], 20)


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

            def fake_run(cmd, cwd, capture_output, text, timeout):
                out_path = Path(cmd[cmd.index("--out") + 1])
                out_path.write_text(json.dumps({"book_urls": ["/a", "/b"]}) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="TimeoutError: net::ERR_CONNECTION_RESET")

            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"), Path("/fake/root"), seed, out_dir, skip_vpn_check=False
                )
        self.assertEqual(status, STATUS_ERROR)
        self.assertIn("ERR_CONNECTION_RESET", error or "")
        self.assertEqual(book_count, 2, "partial output on disk should be counted on error")

    def test_nonzero_exit_without_challenge_marker_is_error_no_partial(self) -> None:
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
        self.assertIsNone(book_count)

    def test_hitting_max_pages_per_run_is_chunked_not_done(self) -> None:
        seed = SeedList(1, "Big_List", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)

            def fake_run(cmd, cwd, capture_output, text, timeout):
                out_path = Path(cmd[cmd.index("--out") + 1])
                lines = "".join(json.dumps({"book_urls": [f"/{i}"]}) + "\n" for i in range(5))
                out_path.write_text(lines, encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"),
                    Path("/fake/root"),
                    seed,
                    out_dir,
                    skip_vpn_check=False,
                    max_pages_per_run=5,
                )
        self.assertEqual(status, STATUS_CHUNKED)
        self.assertIsNone(error)
        self.assertEqual(book_count, 5)

    def test_finishing_before_max_pages_per_run_is_done(self) -> None:
        seed = SeedList(1, "Small_List", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)

            def fake_run(cmd, cwd, capture_output, text, timeout):
                out_path = Path(cmd[cmd.index("--out") + 1])
                out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"),
                    Path("/fake/root"),
                    seed,
                    out_dir,
                    skip_vpn_check=False,
                    max_pages_per_run=50,
                )
        self.assertEqual(status, STATUS_DONE)
        self.assertIsNone(error)
        self.assertEqual(book_count, 1)

    def test_chunked_resume_counts_only_new_pages_this_run(self) -> None:
        """A resumed run must compare pages fetched *this invocation* against
        max_pages_per_run, not the total pages accumulated across all runs —
        otherwise a long list would immediately look 'chunked' forever."""
        seed = SeedList(1, "Resuming_List", "general")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            out_path = out_dir / "1.jsonl"
            existing = "".join(json.dumps({"book_urls": [f"/{i}"]}) + "\n" for i in range(48))
            out_path.write_text(existing, encoding="utf-8")

            def fake_run(cmd, cwd, capture_output, text, timeout):
                # Simulate the harness appending exactly 2 new pages this run.
                new_lines = "".join(json.dumps({"book_urls": [f"/{i}"]}) + "\n" for i in range(50))
                out_path.write_text(new_lines, encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                status, error, _out_path, book_count = run_one_list(
                    Path("/fake/scrape-harness"),
                    Path("/fake/root"),
                    seed,
                    out_dir,
                    skip_vpn_check=False,
                    max_pages_per_run=50,
                )
        self.assertEqual(status, STATUS_DONE, "only 2 new pages were fetched, well under the 50-page chunk cap")
        self.assertIsNone(error)

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
        _write_fake_harness(harness_root)

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

    def test_redo_list_fresh_moves_existing_output_aside(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw = tmp / "raw"
            raw.mkdir()
            out_path = raw / "1.jsonl"
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            cp = Checkpoint(tmp / "checkpoint.sqlite")
            cp.upsert_seed([SeedList(1, "Alpha", "general")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path=str(out_path), book_count=1)

            harness_root = tmp / "harness"
            _write_fake_harness(harness_root)

            def fake_run(cmd, cwd, capture_output, text, timeout):
                out = Path(cmd[cmd.index("--out") + 1])
                self.assertFalse(out.exists(), "--fresh should remove the old output before scraping")
                out.write_text(json.dumps({"book_urls": ["/b"]}) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            args = [
                "--seed-lists",
                str(tmp / "seed.yaml"),
                "--harness-root",
                str(harness_root),
                "--out-dir",
                str(raw),
                "--checkpoint-db",
                str(tmp / "checkpoint.sqlite"),
                "--redo-list",
                "1",
                "--fresh",
                "--shuffle-seed",
                "1",
            ]
            (tmp / "seed.yaml").write_text(SEED_YAML.split("lists:")[0] + "lists:\n  - list_id: 1\n    slug: Alpha\n    genre: general\n")
            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                with patch("tools.scrape_goodreads_lists.time.sleep"):
                    rc = main(args)
            self.assertEqual(rc, 0)
            self.assertEqual(len(list(raw.glob("1.jsonl.bak-*"))), 1)
            self.assertTrue(out_path.exists())

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

    def test_repeated_plain_timeouts_stop_session_before_other_lists(self) -> None:
        """SOFT_BLOCK_THRESHOLD consecutive plain-timeout `error` results on
        one list — accumulated, as in the real incident, across separate
        invocations — must stop the session before it moves on to another
        pending list, without needing a textual challenge marker."""
        calls: list[int] = []

        def fake_run(cmd, cwd, capture_output, text, timeout):
            list_id = int(cmd[cmd.index("--set") + 1].split("=")[1])
            calls.append(list_id)
            if list_id == 2:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="TimeoutError: table.tableList")
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            seed_path = tmp / "seed.yaml"
            seed_path.write_text(SEED_YAML, encoding="utf-8")
            harness_root = tmp / "harness"
            _write_fake_harness(harness_root)
            cp_path = tmp / "checkpoint.sqlite"

            cp = Checkpoint(cp_path)
            cp.upsert_seed([SeedList(1, "Alpha", "general"), SeedList(2, "Beta", "fantasy")])
            # Simulate 2 prior consecutive plain-timeout errors on list 2 from
            # earlier (separate) invocations, cooldown already elapsed.
            cp.conn.execute(
                "UPDATE goodreads_lists SET attempts = 2, consecutive_plain_timeouts = 2 WHERE list_id = 2"
            )
            cp.conn.commit()
            cp.close()

            # shuffle-seed=1 orders [2, 1] for this two-row pending set (see
            # test setup below) — list 2 (the erroring one) runs first.
            args = [
                "--seed-lists",
                str(seed_path),
                "--harness-root",
                str(harness_root),
                "--out-dir",
                str(tmp / "raw"),
                "--checkpoint-db",
                str(cp_path),
                "--shuffle-seed",
                "1",
                "--session-max-lists",
                "10",
            ]
            with patch("tools.scrape_goodreads_lists.subprocess.run", side_effect=fake_run):
                with patch("tools.scrape_goodreads_lists.time.sleep"):
                    rc = main(args)
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [2], "session must stop after list 2's 3rd consecutive plain timeout")

            cp2 = Checkpoint(cp_path)
            row = next(r for r in cp2.rows() if r["list_id"] == 2)
            self.assertEqual(row["consecutive_plain_timeouts"], SOFT_BLOCK_THRESHOLD)
            self.assertEqual(row["status"], STATUS_ERROR)
            cp2.close()

    def test_report_only_does_not_invoke_harness(self) -> None:
        def fake_run(*_a, **_k):
            raise AssertionError("subprocess.run should never be called with --report-only")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc = self._run_main(tmp, SEED_YAML, ["--report-only"], fake_run)
            self.assertEqual(rc, 0)

    def test_skip_deprecated_without_overrides_file_is_a_noop(self) -> None:
        """Missing overrides file + --skip-deprecated must behave exactly like
        the flag was never passed (see tools/catalog/analyze_goodreads_lists.py)."""
        call_count = {"n": 0}

        def fake_run(cmd, cwd, capture_output, text, timeout):
            call_count["n"] += 1
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rc = self._run_main(
                tmp,
                SEED_YAML,
                ["--session-max-lists", "10", "--skip-deprecated", "--overrides", str(tmp / "nonexistent.yaml")],
                fake_run,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(call_count["n"], 2, "both seed lists should still be scraped")

    def test_skip_deprecated_with_empty_overrides_file_is_a_noop(self) -> None:
        call_count = {"n": 0}

        def fake_run(cmd, cwd, capture_output, text, timeout):
            call_count["n"] += 1
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            overrides_path = tmp / "overrides.yaml"
            overrides_path.write_text("overrides: {}\n", encoding="utf-8")
            rc = self._run_main(
                tmp,
                SEED_YAML,
                ["--session-max-lists", "10", "--skip-deprecated", "--overrides", str(overrides_path)],
                fake_run,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(call_count["n"], 2)

    def test_skip_deprecated_with_populated_overrides_skips_deprecated_list(self) -> None:
        scraped_ids: list[int] = []

        def fake_run(cmd, cwd, capture_output, text, timeout):
            list_id = int(cmd[cmd.index("--set") + 1].split("=")[1])
            scraped_ids.append(list_id)
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            overrides_path = tmp / "overrides.yaml"
            overrides_path.write_text(
                "overrides:\n  2:\n    curation_status: deprecated\n    reason: test\n", encoding="utf-8"
            )
            rc = self._run_main(
                tmp,
                SEED_YAML,
                ["--session-max-lists", "10", "--skip-deprecated", "--overrides", str(overrides_path)],
                fake_run,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(scraped_ids, [1], "list 2 is deprecated and must be skipped")

    def test_without_skip_deprecated_flag_overrides_file_is_ignored(self) -> None:
        """The flag must be strictly opt-in: a populated overrides file must
        not affect a run that doesn't pass --skip-deprecated."""
        scraped_ids: list[int] = []

        def fake_run(cmd, cwd, capture_output, text, timeout):
            list_id = int(cmd[cmd.index("--set") + 1].split("=")[1])
            scraped_ids.append(list_id)
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text(json.dumps({"book_urls": ["/a"]}) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            overrides_path = tmp / "overrides.yaml"
            overrides_path.write_text(
                "overrides:\n  2:\n    curation_status: deprecated\n    reason: test\n", encoding="utf-8"
            )
            rc = self._run_main(
                tmp,
                SEED_YAML,
                ["--session-max-lists", "10", "--overrides", str(overrides_path)],
                fake_run,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(scraped_ids), [1, 2])


if __name__ == "__main__":
    unittest.main()
