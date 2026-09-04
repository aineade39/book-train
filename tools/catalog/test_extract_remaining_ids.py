#!/usr/bin/env python3
"""Unit tests for tools/catalog/extract_remaining_ids.py
(run: python tools/catalog/test_extract_remaining_ids.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datetime import datetime, timedelta, timezone  # noqa: E402

from tools.catalog.extract_remaining_ids import (  # noqa: E402
    DEFAULT_GIVE_UP_AFTER,
    build_remaining_queue,
    compute_gave_up,
    compute_remaining,
    count_warning_only_attempts,
    default_book_show_api_paths,
    extract_raw_ids,
    load_attempted_ids,
    load_cooling_ids,
    load_fetched_ids,
    load_fetched_sidecar,
    load_gave_up_ids,
    load_popularity,
    load_popularity_sidecar,
    load_retry_after_sidecar,
    main,
    order_remaining_ids,
    order_remaining_ids_shuffled,
    parse_book_id_from_warning_url,
    update_retry_after,
    write_fetched_sidecar,
    write_ids_remaining,
    write_popularity_sidecar,
)
from tools.catalog.list_show_popularity import BookPopularity  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestExtractRawIds(unittest.TestCase):
    def test_extracts_ids_from_book_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/33.LOTR", "/book/show/74.Hobbit"]}],
            )
            ids = extract_raw_ids(raw_dir)
            self.assertEqual(ids, {33, 74})

    def test_dedupes_across_multiple_list_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [{"book_urls": ["/book/show/33.LOTR"]}])
            _write_jsonl(raw_dir / "2.jsonl", [{"book_urls": ["/book/show/33.LOTR", "/book/show/99.Dune"]}])
            ids = extract_raw_ids(raw_dir)
            self.assertEqual(ids, {33, 99})

    def test_missing_dir_returns_empty_set(self) -> None:
        ids = extract_raw_ids(Path("/nonexistent/raw/dir"))
        self.assertEqual(ids, set())

    def test_skips_malformed_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "1.jsonl").write_text(
                'not json\n{"book_urls": ["/book/show/33.LOTR"]}\n', encoding="utf-8"
            )
            ids = extract_raw_ids(raw_dir)
            self.assertEqual(ids, {33})

    def test_ignores_urls_without_book_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_jsonl(raw_dir / "1.jsonl", [{"book_urls": ["/author/show/1.Someone"]}])
            ids = extract_raw_ids(raw_dir)
            self.assertEqual(ids, set())


class TestFetchedSidecar(unittest.TestCase):
    def test_write_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "fetched_ids.txt"
            write_fetched_sidecar(sidecar, {33, 74, 99})
            self.assertEqual(load_fetched_sidecar(sidecar), {33, 74, 99})

    def test_load_missing_sidecar_returns_empty(self) -> None:
        self.assertEqual(load_fetched_sidecar(Path("/nonexistent/fetched_ids.txt")), set())

    def test_write_empty_set_produces_readable_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "fetched_ids.txt"
            write_fetched_sidecar(sidecar, set())
            self.assertEqual(load_fetched_sidecar(sidecar), set())

    def test_load_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "fetched_ids.txt"
            sidecar.write_text("33\nnot-a-number\n74\n", encoding="utf-8")
            self.assertEqual(load_fetched_sidecar(sidecar), {33, 74})


class TestLoadFetchedIds(unittest.TestCase):
    def test_bootstraps_sidecar_from_jsonl_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api_path = Path(tmp) / "book_show_api.jsonl"
            sidecar_path = Path(tmp) / "fetched_ids.txt"
            _write_jsonl(api_path, [{"legacy_id": 33, "isbn13": "x"}, {"legacy_id": 74, "isbn13": "y"}])

            ids = load_fetched_ids(api_path, sidecar_path)
            self.assertEqual(ids, {33, 74})
            self.assertTrue(sidecar_path.exists())
            self.assertEqual(load_fetched_sidecar(sidecar_path), {33, 74})

    def test_uses_existing_sidecar_without_rescanning_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api_path = Path(tmp) / "book_show_api.jsonl"
            sidecar_path = Path(tmp) / "fetched_ids.txt"
            _write_jsonl(api_path, [{"legacy_id": 33, "isbn13": "x"}])
            time.sleep(0.01)
            write_fetched_sidecar(sidecar_path, {33, 999})  # deliberately stale/different from JSONL

            # Sidecar is newer than the JSONL -> trusted as-is, no rebuild.
            ids = load_fetched_ids(api_path, sidecar_path)
            self.assertEqual(ids, {33, 999})

    def test_rebuilds_sidecar_when_jsonl_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api_path = Path(tmp) / "book_show_api.jsonl"
            sidecar_path = Path(tmp) / "fetched_ids.txt"
            write_fetched_sidecar(sidecar_path, {1})
            time.sleep(0.01)
            _write_jsonl(api_path, [{"legacy_id": 33, "isbn13": "x"}, {"legacy_id": 74, "isbn13": "y"}])

            ids = load_fetched_ids(api_path, sidecar_path)
            self.assertEqual(ids, {33, 74})

    def test_missing_jsonl_returns_empty_regardless_of_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api_path = Path(tmp) / "book_show_api.jsonl"
            sidecar_path = Path(tmp) / "fetched_ids.txt"
            write_fetched_sidecar(sidecar_path, {1, 2, 3})

            ids = load_fetched_ids(api_path, sidecar_path)
            self.assertEqual(ids, set())

    def test_skips_records_without_legacy_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api_path = Path(tmp) / "book_show_api.jsonl"
            sidecar_path = Path(tmp) / "fetched_ids.txt"
            _write_jsonl(
                api_path,
                [
                    {"legacy_id": 33, "isbn13": "x"},
                    {"_record_name": "book_page", "_scrape_warning": "blocked_suspected"},
                ],
            )
            ids = load_fetched_ids(api_path, sidecar_path)
            self.assertEqual(ids, {33})


class TestDefaultBookShowApiPaths(unittest.TestCase):
    def test_globs_all_shards_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_jsonl(tmp_path / "book_show_api.jsonl", [])
            _write_jsonl(tmp_path / "book_show_api.batch2.jsonl", [])
            _write_jsonl(tmp_path / "book_show_api.part1.jsonl", [])
            _write_jsonl(tmp_path / "unrelated.jsonl", [])

            paths = default_book_show_api_paths(tmp_path)

            self.assertEqual([p.name for p in paths], sorted(
                ["book_show_api.jsonl", "book_show_api.batch2.jsonl", "book_show_api.part1.jsonl"]
            ))

    def test_no_shards_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(default_book_show_api_paths(Path(tmp)), [])


class TestLoadFetchedIdsMultiFile(unittest.TestCase):
    def test_unions_legacy_ids_across_multiple_shard_files(self) -> None:
        """A book_show_api run interrupted and resumed under a different --out
        name (e.g. book_show_api.part1.jsonl / book_show_api.batch2.jsonl) must
        not strand either shard's progress."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shard_a = tmp_path / "book_show_api.jsonl"
            shard_b = tmp_path / "book_show_api.batch2.jsonl"
            sidecar_path = tmp_path / "fetched_ids.txt"
            _write_jsonl(shard_a, [{"legacy_id": 33, "isbn13": "x"}])
            _write_jsonl(shard_b, [{"legacy_id": 74, "isbn13": "y"}])

            ids = load_fetched_ids([shard_a, shard_b], sidecar_path)

            self.assertEqual(ids, {33, 74})

    def test_single_path_still_works_unwrapped(self) -> None:
        """Backward-compatible: a bare Path (not a list) is still accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shard = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "fetched_ids.txt"
            _write_jsonl(shard, [{"legacy_id": 33, "isbn13": "x"}])

            self.assertEqual(load_fetched_ids(shard, sidecar_path), {33})

    def test_missing_shards_return_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ids = load_fetched_ids(
                [tmp_path / "missing1.jsonl", tmp_path / "missing2.jsonl"], tmp_path / "fetched_ids.txt"
            )
            self.assertEqual(ids, set())


class TestDefaultGiveUpAfter(unittest.TestCase):
    def test_default_is_three(self) -> None:
        self.assertEqual(DEFAULT_GIVE_UP_AFTER, 3)


class TestParseBookIdFromWarningUrl(unittest.TestCase):
    def test_parses_next_data_book_show_url(self) -> None:
        url = "https://www.goodreads.com/_next/data/4uaR8Y6o0sn5STIiRSKQ4/book/show/33.json"
        self.assertEqual(parse_book_id_from_warning_url(url), 33)

    def test_returns_none_for_non_matching_url(self) -> None:
        self.assertIsNone(parse_book_id_from_warning_url("https://www.goodreads.com/author/show/1.json"))

    def test_returns_none_for_none(self) -> None:
        self.assertIsNone(parse_book_id_from_warning_url(None))

    def test_returns_none_for_empty_string(self) -> None:
        self.assertIsNone(parse_book_id_from_warning_url(""))


class TestCountWarningOnlyAttempts(unittest.TestCase):
    def test_tallies_warnings_per_book_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            url_74 = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            shard_a = tmp_path / "book_show_api.jsonl"
            shard_b = tmp_path / "book_show_api.batch2.jsonl"
            _write_jsonl(shard_a, [{"_scrape_warning": "incomplete_record", "_url": url_74}])
            _write_jsonl(
                shard_b,
                [
                    {"_scrape_warning": "incomplete_record", "_url": url_74},
                    {"_scrape_warning": "incomplete_record", "_url": url_74},
                ],
            )

            counts = count_warning_only_attempts([shard_a, shard_b])
            self.assertEqual(counts, {74: 3})

    def test_honors_existing_attempt_count_field(self) -> None:
        """A merge_book_show_api.py-aggregated record's _attempt_count is
        summed rather than counted as a single attempt."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(path, [{"_scrape_warning": "incomplete_record", "_url": url, "_attempt_count": 5}])

            self.assertEqual(count_warning_only_attempts([path]), {74: 5})

    def test_ignores_records_with_legacy_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [{"legacy_id": 33, "isbn13": "x"}])

            self.assertEqual(count_warning_only_attempts([path]), {})

    def test_ignores_warnings_with_unparseable_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            _write_jsonl(path, [{"_scrape_warning": "blocked_suspected", "_url": "https://example.com/unrelated"}])

            self.assertEqual(count_warning_only_attempts([path]), {})

    def test_missing_shard_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(count_warning_only_attempts([Path(tmp) / "missing.jsonl"]), {})


class TestComputeGaveUp(unittest.TestCase):
    def test_excludes_ids_below_threshold(self) -> None:
        self.assertEqual(compute_gave_up({74: 2}, set(), give_up_after=3), set())

    def test_includes_ids_at_or_above_threshold(self) -> None:
        self.assertEqual(compute_gave_up({74: 3, 99: 4}, set(), give_up_after=3), {74, 99})

    def test_success_overrides_prior_warnings(self) -> None:
        """A book that ever produced a legacy_id is never given up, no
        matter how many warnings it accumulated before that."""
        self.assertEqual(compute_gave_up({74: 10}, fetched_ids={74}, give_up_after=3), set())


class TestLoadGaveUpIds(unittest.TestCase):
    def test_rebuilds_from_jsonl_when_sidecar_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(api_path, [{"_scrape_warning": "incomplete_record", "_url": url}] * 3)

            gave_up = load_gave_up_ids(api_path, sidecar_path, fetched_ids=set(), give_up_after=3)

            self.assertEqual(gave_up, {74})
            self.assertTrue(sidecar_path.exists())
            self.assertEqual(load_fetched_sidecar(sidecar_path), {74})

    def test_uses_existing_sidecar_without_rescanning_when_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            _write_jsonl(api_path, [])
            time.sleep(0.01)
            write_fetched_sidecar(sidecar_path, {74})  # deliberately not recomputed from api_path

            gave_up = load_gave_up_ids(api_path, sidecar_path, fetched_ids=set(), give_up_after=3)
            self.assertEqual(gave_up, {74})

    def test_rebuilds_when_jsonl_is_newer_than_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            write_fetched_sidecar(sidecar_path, set())
            time.sleep(0.01)
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(api_path, [{"_scrape_warning": "incomplete_record", "_url": url}] * 3)

            gave_up = load_gave_up_ids(api_path, sidecar_path, fetched_ids=set(), give_up_after=3)
            self.assertEqual(gave_up, {74})

    def test_missing_jsonl_returns_empty_regardless_of_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            write_fetched_sidecar(sidecar_path, {74})

            gave_up = load_gave_up_ids(api_path, sidecar_path, fetched_ids=set(), give_up_after=3)
            self.assertEqual(gave_up, set())

    def test_fetched_ids_prevent_give_up_on_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(api_path, [{"_scrape_warning": "incomplete_record", "_url": url}] * 5)

            gave_up = load_gave_up_ids(api_path, sidecar_path, fetched_ids={74}, give_up_after=3)
            self.assertEqual(gave_up, set())


class TestComputeRemaining(unittest.TestCase):
    def test_subtracts_fetched_from_raw(self) -> None:
        self.assertEqual(compute_remaining({1, 2, 3}, {2}), {1, 3})

    def test_subtracts_gave_up_ids_too(self) -> None:
        self.assertEqual(compute_remaining({1, 2, 3}, {2}, gave_up_ids={3}), {1})

    def test_default_gave_up_ids_is_empty(self) -> None:
        self.assertEqual(compute_remaining({1, 2, 3}, {2}), compute_remaining({1, 2, 3}, {2}, gave_up_ids=set()))

    def test_all_fetched_returns_empty(self) -> None:
        self.assertEqual(compute_remaining({1, 2}, {1, 2, 3}), set())

    def test_none_fetched_returns_all_raw(self) -> None:
        self.assertEqual(compute_remaining({1, 2}, set()), {1, 2})


class TestOrderRemainingIds(unittest.TestCase):
    def test_never_tried_ids_come_before_retries(self) -> None:
        ordered = order_remaining_ids({1, 11, 365, 408}, attempted_ids={1, 11}, popularity={})
        self.assertEqual(set(ordered[:2]), {365, 408})
        self.assertEqual(ordered[2:], [1, 11])

    def test_all_retries_stays_sorted_at_tail(self) -> None:
        self.assertEqual(order_remaining_ids({50, 11, 28}, {11, 28, 50}, popularity={}), [11, 28, 50])

    def test_higher_ratings_count_sorts_first_within_never_tried(self) -> None:
        popularity = {
            1: BookPopularity(ratings_count=10),
            2: BookPopularity(ratings_count=1_000_000),
            3: BookPopularity(ratings_count=500),
        }
        ordered = order_remaining_ids({1, 2, 3}, attempted_ids=set(), popularity=popularity)
        self.assertEqual(ordered, [2, 3, 1])

    def test_missing_popularity_signal_sorts_to_tail_of_bucket(self) -> None:
        popularity = {1: BookPopularity(ratings_count=500)}
        ordered = order_remaining_ids({1, 2}, attempted_ids=set(), popularity=popularity)
        self.assertEqual(ordered, [1, 2])

    def test_popularity_ordering_applies_within_retries_too(self) -> None:
        popularity = {74: BookPopularity(ratings_count=10), 99: BookPopularity(ratings_count=500)}
        ordered = order_remaining_ids({74, 99}, attempted_ids={74, 99}, popularity=popularity)
        self.assertEqual(ordered, [99, 74])

    def test_rebuilds_when_sidecar_is_empty_but_raw_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            sidecar_path = tmp_path / "book_popularity.json"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR"],
                        "rating_texts": ["4.55 avg rating — 745,415 ratings"],
                        "score_texts": [""],
                        "vote_texts": [""],
                    }
                ],
            )
            time.sleep(0.01)
            sidecar_path.write_text('{"books": {}}', encoding="utf-8")

            popularity = load_popularity(raw_dir, sidecar_path)
            self.assertEqual(popularity[33].ratings_count, 745415)

    def test_no_popularity_signal_ties_break_by_book_id(self) -> None:
        ordered = order_remaining_ids({408, 365}, attempted_ids=set(), popularity={})
        self.assertEqual(ordered, [365, 408])


class TestOrderRemainingIdsShuffled(unittest.TestCase):
    def test_never_tried_ids_come_before_retries(self) -> None:
        ordered = order_remaining_ids_shuffled({1, 11, 365, 408}, attempted_ids={1, 11})
        self.assertEqual(set(ordered[:2]), {365, 408})
        self.assertEqual(ordered[2:], [1, 11])

    def test_all_retries_stays_sorted_at_tail(self) -> None:
        self.assertEqual(order_remaining_ids_shuffled({50, 11, 28}, {11, 28, 50}), [11, 28, 50])


class TestPopularitySidecar(unittest.TestCase):
    def test_write_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "book_popularity.json"
            popularity = {33: BookPopularity(ratings_count=745415, list_appearances=2, list_score_sum=100)}
            write_popularity_sidecar(sidecar, popularity)
            self.assertEqual(load_popularity_sidecar(sidecar), popularity)

    def test_load_missing_sidecar_returns_empty(self) -> None:
        self.assertEqual(load_popularity_sidecar(Path("/nonexistent/book_popularity.json")), {})

    def test_load_malformed_json_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "book_popularity.json"
            sidecar.write_text("not json", encoding="utf-8")
            self.assertEqual(load_popularity_sidecar(sidecar), {})


class TestLoadPopularity(unittest.TestCase):
    def test_bootstraps_sidecar_from_raw_dir_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            sidecar_path = tmp_path / "book_popularity.json"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR"],
                        "rating_texts": ["4.55 avg rating — 745,415 ratings"],
                        "score_texts": [""],
                        "vote_texts": [""],
                    }
                ],
            )

            popularity = load_popularity(raw_dir, sidecar_path)

            self.assertEqual(popularity[33].ratings_count, 745415)
            self.assertTrue(sidecar_path.exists())
            self.assertEqual(load_popularity_sidecar(sidecar_path)[33].ratings_count, 745415)

    def test_uses_existing_sidecar_without_rescanning_raw_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            sidecar_path = tmp_path / "book_popularity.json"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/33.LOTR"], "rating_texts": ["4.20 avg rating — 10 ratings"],
                  "score_texts": [""], "vote_texts": [""]}],
            )
            time.sleep(0.01)
            write_popularity_sidecar(sidecar_path, {33: BookPopularity(ratings_count=999)})  # deliberately stale

            popularity = load_popularity(raw_dir, sidecar_path)
            self.assertEqual(popularity[33].ratings_count, 999)

    def test_rebuilds_when_raw_dir_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            sidecar_path = tmp_path / "book_popularity.json"
            write_popularity_sidecar(sidecar_path, {33: BookPopularity(ratings_count=1)})
            time.sleep(0.01)
            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/33.LOTR"], "rating_texts": ["4.20 avg rating — 745,415 ratings"],
                  "score_texts": [""], "vote_texts": [""]}],
            )

            popularity = load_popularity(raw_dir, sidecar_path)
            self.assertEqual(popularity[33].ratings_count, 745415)

    def test_missing_raw_dir_falls_back_to_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sidecar_path = tmp_path / "book_popularity.json"
            write_popularity_sidecar(sidecar_path, {33: BookPopularity(ratings_count=5)})

            popularity = load_popularity(tmp_path / "raw", sidecar_path)
            self.assertEqual(popularity[33].ratings_count, 5)


class TestLoadAttemptedIds(unittest.TestCase):
    def test_unions_warning_book_ids_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            url = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            _write_jsonl(tmp_path / "a.jsonl", [{"_scrape_warning": "incomplete_record", "_url": url}])
            _write_jsonl(tmp_path / "b.jsonl", [{"_scrape_warning": "blocked_suspected", "_url": url}])
            self.assertEqual(load_attempted_ids([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]), {74})


class TestWriteIdsRemaining(unittest.TestCase):
    def test_writes_one_id_per_line_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ids_remaining.txt"
            write_ids_remaining(out_path, {74, 33, 99})
            self.assertEqual(out_path.read_text(encoding="utf-8").splitlines(), ["33", "74", "99"])

    def test_preserves_explicit_order_for_never_tried_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ids_remaining.txt"
            write_ids_remaining(out_path, [365, 408, 1, 11])
            self.assertEqual(out_path.read_text(encoding="utf-8").splitlines(), ["365", "408", "1", "11"])

    def test_empty_set_writes_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ids_remaining.txt"
            write_ids_remaining(out_path, set())
            self.assertEqual(out_path.read_text(encoding="utf-8"), "")


class TestMainEndToEnd(unittest.TestCase):
    def test_full_pipeline_produces_expected_remaining_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "fetched_ids.txt"
            gave_up_sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            out_path = tmp_path / "ids_remaining.txt"

            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/33.LOTR", "/book/show/74.Hobbit", "/book/show/99.Dune"]}],
            )
            _write_jsonl(api_path, [{"legacy_id": 33, "isbn13": "x"}])

            rc = main(
                [
                    "--raw-dir",
                    str(raw_dir),
                    "--book-show-api",
                    str(api_path),
                    "--fetched-sidecar",
                    str(sidecar_path),
                    "--gave-up-sidecar",
                    str(gave_up_sidecar_path),
                    "--retry-after-sidecar",
                    str(tmp_path / "book_show_api_retry_after.json"),
                    "--retry-cooldown-hours",
                    "0",
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(set(out_path.read_text(encoding="utf-8").split()), {"74", "99"})

    def test_rerun_after_fetching_more_shrinks_remaining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "fetched_ids.txt"
            gave_up_sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            out_path = tmp_path / "ids_remaining.txt"

            _write_jsonl(raw_dir / "1.jsonl", [{"book_urls": ["/book/show/33.LOTR", "/book/show/74.Hobbit"]}])
            _write_jsonl(api_path, [])

            retry_sidecar = str(tmp_path / "book_show_api_retry_after.json")
            main(["--raw-dir", str(raw_dir), "--book-show-api", str(api_path),
                  "--fetched-sidecar", str(sidecar_path), "--gave-up-sidecar", str(gave_up_sidecar_path),
                  "--retry-after-sidecar", retry_sidecar, "--retry-cooldown-hours", "0",
                  "--out", str(out_path)])
            self.assertEqual(set(out_path.read_text(encoding="utf-8").split()), {"33", "74"})

            # Simulate a partial book_show_api run having fetched book 33.
            time.sleep(0.01)
            _write_jsonl(api_path, [{"legacy_id": 33, "isbn13": "x"}])
            main(["--raw-dir", str(raw_dir), "--book-show-api", str(api_path),
                  "--fetched-sidecar", str(sidecar_path), "--gave-up-sidecar", str(gave_up_sidecar_path),
                  "--retry-after-sidecar", retry_sidecar, "--retry-cooldown-hours", "0",
                  "--out", str(out_path)])
            self.assertEqual(set(out_path.read_text(encoding="utf-8").split()), {"74"})

    def test_no_raw_files_and_no_api_output_produces_empty_remaining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rc = main(
                [
                    "--raw-dir", str(tmp_path / "raw"),
                    "--book-show-api", str(tmp_path / "book_show_api.jsonl"),
                    "--fetched-sidecar", str(tmp_path / "fetched_ids.txt"),
                    "--gave-up-sidecar", str(tmp_path / "book_show_api_gave_up.txt"),
                    "--retry-after-sidecar", str(tmp_path / "book_show_api_retry_after.json"),
                    "--out", str(tmp_path / "ids_remaining.txt"),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual((tmp_path / "ids_remaining.txt").read_text(encoding="utf-8"), "")

    def test_books_that_never_succeed_are_given_up_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "fetched_ids.txt"
            gave_up_sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            out_path = tmp_path / "ids_remaining.txt"

            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/33.LOTR", "/book/show/74.Hobbit", "/book/show/99.Dune"]}],
            )
            url_74 = "https://www.goodreads.com/_next/data/abc/book/show/74.json"
            url_99 = "https://www.goodreads.com/_next/data/abc/book/show/99.json"
            _write_jsonl(
                api_path,
                [{"legacy_id": 33, "isbn13": "x"}]
                + [{"_scrape_warning": "incomplete_record", "_url": url_74}] * 3
                + [{"_scrape_warning": "incomplete_record", "_url": url_99}],
            )

            rc = main(
                [
                    "--raw-dir", str(raw_dir),
                    "--book-show-api", str(api_path),
                    "--fetched-sidecar", str(sidecar_path),
                    "--gave-up-sidecar", str(gave_up_sidecar_path),
                    "--give-up-after", "3",
                    "--retry-after-sidecar", str(tmp_path / "book_show_api_retry_after.json"),
                    "--retry-cooldown-hours", "0",
                    "--out", str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(out_path.read_text(encoding="utf-8").splitlines(), ["99"])
            self.assertEqual(load_fetched_sidecar(gave_up_sidecar_path), {74})

    def test_popularity_order_places_higher_ratings_count_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            out_path = tmp_path / "ids_remaining.txt"

            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR", "/book/show/74.Hobbit"],
                        "rating_texts": ["4.20 avg rating — 100 ratings", "4.55 avg rating — 745,415 ratings"],
                        "score_texts": ["", ""],
                        "vote_texts": ["", ""],
                    }
                ],
            )
            _write_jsonl(api_path, [])

            rc = main(
                [
                    "--raw-dir", str(raw_dir),
                    "--book-show-api", str(api_path),
                    "--fetched-sidecar", str(tmp_path / "fetched_ids.txt"),
                    "--gave-up-sidecar", str(tmp_path / "book_show_api_gave_up.txt"),
                    "--popularity-sidecar", str(tmp_path / "book_popularity.json"),
                    "--retry-after-sidecar", str(tmp_path / "book_show_api_retry_after.json"),
                    "--out", str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(out_path.read_text(encoding="utf-8").splitlines(), ["74", "33"])

    def test_order_shuffle_flag_still_produces_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            out_path = tmp_path / "ids_remaining.txt"

            _write_jsonl(raw_dir / "1.jsonl", [{"book_urls": ["/book/show/33.LOTR", "/book/show/74.Hobbit"]}])
            _write_jsonl(api_path, [])

            rc = main(
                [
                    "--raw-dir", str(raw_dir),
                    "--book-show-api", str(api_path),
                    "--fetched-sidecar", str(tmp_path / "fetched_ids.txt"),
                    "--gave-up-sidecar", str(tmp_path / "book_show_api_gave_up.txt"),
                    "--order", "shuffle",
                    "--retry-after-sidecar", str(tmp_path / "book_show_api_retry_after.json"),
                    "--out", str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(set(out_path.read_text(encoding="utf-8").split()), {"33", "74"})

    def test_never_tried_ids_are_written_before_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            sidecar_path = tmp_path / "fetched_ids.txt"
            gave_up_sidecar_path = tmp_path / "book_show_api_gave_up.txt"
            out_path = tmp_path / "ids_remaining.txt"
            url_11 = "https://www.goodreads.com/_next/data/abc/book/show/11.json"

            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/11.Hard", "/book/show/365.Fresh", "/book/show/408.Fresh"]}],
            )
            _write_jsonl(api_path, [{"_scrape_warning": "incomplete_record", "_url": url_11}])

            rc = main(
                [
                    "--raw-dir", str(raw_dir),
                    "--book-show-api", str(api_path),
                    "--fetched-sidecar", str(sidecar_path),
                    "--gave-up-sidecar", str(gave_up_sidecar_path),
                    "--retry-after-sidecar", str(tmp_path / "book_show_api_retry_after.json"),
                    "--retry-cooldown-hours", "0",
                    "--out", str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(set(lines), {"365", "408", "11"})
            self.assertEqual(lines[-1], "11")
            self.assertIn("365", lines[:2])
            self.assertIn("408", lines[:2])


class TestBuildRemainingQueue(unittest.TestCase):
    def _paths(self, tmp: Path) -> dict[str, Path]:
        return {
            "raw_dir": tmp / "raw",
            "api": tmp / "book_show_api.jsonl",
            "fetched": tmp / "fetched_ids.txt",
            "gave_up": tmp / "book_show_api_gave_up.txt",
            "popularity": tmp / "book_popularity.json",
            "retry_after": tmp / "book_show_api_retry_after.json",
        }

    def test_persist_false_withholds_cooling_ids_without_stamping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
            url_11 = "https://www.goodreads.com/_next/data/abc/book/show/11.json"
            _write_jsonl(
                paths["raw_dir"] / "1.jsonl",
                [{"book_urls": ["/book/show/11.Hard", "/book/show/365.Fresh"]}],
            )
            _write_jsonl(paths["api"], [{"_scrape_warning": "incomplete_record", "_url": url_11}])
            update_retry_after({11: 1}, paths["retry_after"], 24.0, now=now)
            before = paths["retry_after"].read_text(encoding="utf-8")

            queue = build_remaining_queue(
                paths["raw_dir"],
                [paths["api"]],
                fetched_sidecar=paths["fetched"],
                gave_up_sidecar=paths["gave_up"],
                popularity_sidecar=paths["popularity"],
                retry_after_sidecar=paths["retry_after"],
                persist=False,
                now=now,
            )
            self.assertEqual(queue.remaining, {365})
            self.assertEqual(queue.cooling_ids, {11})
            self.assertEqual(queue.ordered, [365])
            self.assertEqual(paths["retry_after"].read_text(encoding="utf-8"), before)

    def test_persist_false_uses_cached_popularity_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            _write_jsonl(
                paths["raw_dir"] / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR", "/book/show/99.Dune"],
                        "rating_texts": ["4.5 avg rating — 10 ratings", "4.6 avg rating — 9 ratings"],
                        "score_texts": ["", ""],
                        "vote_texts": ["", ""],
                    }
                ],
            )
            _write_jsonl(paths["api"], [])
            write_popularity_sidecar(
                paths["popularity"],
                {
                    33: BookPopularity(ratings_count=10),
                    99: BookPopularity(ratings_count=500_000),
                },
            )
            time.sleep(0.01)
            sidecar_before = paths["popularity"].read_text(encoding="utf-8")

            queue = build_remaining_queue(
                paths["raw_dir"],
                [paths["api"]],
                fetched_sidecar=paths["fetched"],
                gave_up_sidecar=paths["gave_up"],
                popularity_sidecar=paths["popularity"],
                retry_after_sidecar=paths["retry_after"],
                persist=False,
            )
            self.assertEqual(queue.ordered, [99, 33])
            self.assertEqual(queue.popularity[99].ratings_count, 500_000)
            self.assertEqual(paths["popularity"].read_text(encoding="utf-8"), sidecar_before)

    def test_persist_true_matches_main_cooling_and_writes_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            url_11 = "https://www.goodreads.com/_next/data/abc/book/show/11.json"
            _write_jsonl(
                paths["raw_dir"] / "1.jsonl",
                [{"book_urls": ["/book/show/11.Hard", "/book/show/365.Fresh"]}],
            )
            _write_jsonl(paths["api"], [{"_scrape_warning": "incomplete_record", "_url": url_11}])

            queue = build_remaining_queue(
                paths["raw_dir"],
                [paths["api"]],
                fetched_sidecar=paths["fetched"],
                gave_up_sidecar=paths["gave_up"],
                popularity_sidecar=paths["popularity"],
                retry_after_sidecar=paths["retry_after"],
                persist=True,
            )
            self.assertEqual(queue.remaining, {365})
            self.assertEqual(queue.cooling_ids, {11})
            self.assertIn(11, load_retry_after_sidecar(paths["retry_after"]))


class TestRetryCooldown(unittest.TestCase):
    def test_new_warning_withholds_id_until_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "book_show_api_retry_after.json"
            now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
            cooling = update_retry_after({11: 1}, sidecar, 24.0, now=now)
            self.assertEqual(cooling, {11})
            stored = load_retry_after_sidecar(sidecar)
            self.assertEqual(stored[11]["attempts"], 1)
            self.assertEqual(load_cooling_ids(sidecar, now=now), {11})
            self.assertEqual(
                load_cooling_ids(sidecar, now=now + timedelta(hours=24)),
                set(),
            )

    def test_same_attempt_count_does_not_refresh_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "book_show_api_retry_after.json"
            first = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
            update_retry_after({11: 1}, sidecar, 24.0, now=first)
            later = first + timedelta(hours=1)
            update_retry_after({11: 1}, sidecar, 24.0, now=later)
            stored = load_retry_after_sidecar(sidecar)
            self.assertEqual(stored[11]["retry_after"], (first + timedelta(hours=24)).isoformat(timespec="seconds"))

    def test_increased_attempt_count_restamps_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "book_show_api_retry_after.json"
            first = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
            update_retry_after({11: 1}, sidecar, 24.0, now=first)
            later = first + timedelta(hours=1)
            cooling = update_retry_after({11: 2}, sidecar, 24.0, now=later)
            self.assertEqual(cooling, {11})
            stored = load_retry_after_sidecar(sidecar)
            self.assertEqual(stored[11]["attempts"], 2)
            self.assertEqual(stored[11]["retry_after"], (later + timedelta(hours=24)).isoformat(timespec="seconds"))

    def test_main_withholds_cooling_retry_and_keeps_never_tried_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            api_path = tmp_path / "book_show_api.jsonl"
            out_path = tmp_path / "ids_remaining.txt"
            url_11 = "https://www.goodreads.com/_next/data/abc/book/show/11.json"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [{"book_urls": ["/book/show/11.Hard", "/book/show/365.Fresh"]}],
            )
            _write_jsonl(api_path, [{"_scrape_warning": "incomplete_record", "_url": url_11}])

            rc = main(
                [
                    "--raw-dir", str(raw_dir),
                    "--book-show-api", str(api_path),
                    "--fetched-sidecar", str(tmp_path / "fetched_ids.txt"),
                    "--gave-up-sidecar", str(tmp_path / "book_show_api_gave_up.txt"),
                    "--retry-after-sidecar", str(tmp_path / "book_show_api_retry_after.json"),
                    "--retry-cooldown-hours", "24",
                    "--out", str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(out_path.read_text(encoding="utf-8").splitlines(), ["365"])


if __name__ == "__main__":
    unittest.main()
