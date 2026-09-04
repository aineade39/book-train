#!/usr/bin/env python3
"""Unit tests for tools/catalog/bench_book_show_api_pacing.py
(run: python tools/catalog/test_bench_book_show_api_pacing.py).

Only exercises pure functions -- no live scrape-harness invocation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.bench_book_show_api_pacing import (  # noqa: E402
    SiteLockedError,
    TierResult,
    check_site_not_locked,
    parse_tier_output,
    select_sample,
    verdict,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestSelectSample(unittest.TestCase):
    def test_only_picks_ids_with_legacy_id_and_isbn13(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(
                path,
                [
                    {"legacy_id": 1, "isbn13": "9780000000001"},
                    {"legacy_id": 2, "isbn13": None},
                    {"_scrape_warning": "incomplete_record", "_url": "https://x/book/show/3.json"},
                ],
            )
            sample = select_sample([path], sample_size=20, seed=42)
            self.assertEqual(sample, [1])

    def test_deterministic_given_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [{"legacy_id": i, "isbn13": f"978000000{i:04d}"} for i in range(50)])

            first = select_sample([path], sample_size=10, seed=7)
            second = select_sample([path], sample_size=10, seed=7)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 10)

    def test_different_seeds_can_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [{"legacy_id": i, "isbn13": f"978000000{i:04d}"} for i in range(50)])

            first = select_sample([path], sample_size=10, seed=1)
            second = select_sample([path], sample_size=10, seed=2)
            self.assertNotEqual(first, second)

    def test_returns_all_candidates_when_fewer_than_sample_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [{"legacy_id": 1, "isbn13": "9780000000001"}, {"legacy_id": 2, "isbn13": "9780000000002"}])
            sample = select_sample([path], sample_size=20, seed=42)
            self.assertEqual(sample, [1, 2])

    def test_missing_path_yields_empty_sample(self) -> None:
        sample = select_sample([Path("/tmp/does-not-exist-book-show-api.jsonl")], sample_size=20, seed=42)
        self.assertEqual(sample, [])


class TestParseTierOutput(unittest.TestCase):
    def test_tallies_success_incomplete_and_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tier.jsonl"
            _write_jsonl(
                path,
                [
                    {"legacy_id": 1, "isbn13": "9780000000001"},
                    {"legacy_id": 2, "isbn13": "9780000000002"},
                    {"_scrape_warning": "incomplete_record", "_url": "https://x/book/show/3.json"},
                    {"_scrape_warning": "blocked_suspected", "_url": "https://x/book/show/4.json"},
                ],
            )
            successes, incomplete, blocked = parse_tier_output(path)
            self.assertEqual((successes, incomplete, blocked), (2, 1, 1))

    def test_missing_output_file_is_all_zero(self) -> None:
        successes, incomplete, blocked = parse_tier_output(Path("/tmp/does-not-exist-tier.jsonl"))
        self.assertEqual((successes, incomplete, blocked), (0, 0, 0))

    def test_legacy_id_without_isbn13_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tier.jsonl"
            _write_jsonl(
                path,
                [
                    {"legacy_id": 2, "isbn13": None},
                    {"legacy_id": 3},
                ],
            )
            successes, incomplete, blocked = parse_tier_output(path)
            self.assertEqual((successes, incomplete, blocked), (0, 0, 0))


class TestVerdict(unittest.TestCase):
    def _tier(self, successes: int, total: int, blocked: int = 0) -> TierResult:
        return TierResult(
            name="t", delay_ms_min=0, delay_ms_max=0, total=total, successes=successes,
            incomplete=total - successes - blocked, blocked=blocked, elapsed_s=1.0,
        )

    def test_fast_ok_when_success_rate_matches_baseline(self) -> None:
        baseline = self._tier(successes=18, total=20)
        fast = self._tier(successes=18, total=20)
        self.assertTrue(verdict(baseline, fast))

    def test_fast_ok_within_tolerance(self) -> None:
        baseline = self._tier(successes=20, total=20)
        fast = self._tier(successes=19, total=20)
        self.assertTrue(verdict(baseline, fast, tolerance=0.05))

    def test_fast_fail_when_success_rate_drops_too_much(self) -> None:
        baseline = self._tier(successes=18, total=20)
        fast = self._tier(successes=10, total=20)
        self.assertFalse(verdict(baseline, fast))

    def test_fast_fail_on_any_bot_wall_hit_even_with_good_success_rate(self) -> None:
        baseline = self._tier(successes=18, total=20)
        fast = self._tier(successes=18, total=20, blocked=1)
        self.assertFalse(verdict(baseline, fast))


class TestCheckSiteNotLocked(unittest.TestCase):
    def test_no_lock_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            check_site_not_locked(Path(tmp), site_id="goodreads")

    def test_held_lock_raises(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as tmp:
            harness_root = Path(tmp)
            lock_path = harness_root / ".data" / "locks" / "goodreads.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = lock_path.open("a+")
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(SiteLockedError):
                    check_site_not_locked(harness_root, site_id="goodreads")
            finally:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
                fd.close()


if __name__ == "__main__":
    unittest.main()
