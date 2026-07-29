#!/usr/bin/env python3
"""Unit tests for tools/catalog/analyze_goodreads_lists.py
(run: python tools/catalog/test_analyze_goodreads_lists.py).

No network access, no real scrape-harness data: raw list_show fixtures are
written as plain JSONL and a checkpoint DB is built via the real
`scrape_goodreads_lists.Checkpoint` class so the schema this tool reads
never silently drifts from what the scraper actually writes.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.analyze_goodreads_lists import (  # noqa: E402
    DONE_STATUS,
    OUTCOME_DEPRECATE,
    OUTCOME_KEEP,
    ListData,
    Recommendation,
    SeedListInfo,
    apply_deprecations,
    build_report,
    containment,
    evaluate_lists,
    greedy_set_cover,
    jaccard,
    load_checkpoint_status,
    load_list_data,
    load_overrides,
    load_seed_lists,
    top_n_book_ids,
)
from tools.scrape_goodreads_lists import Checkpoint, SeedList, STATUS_DONE, STATUS_PENDING  # noqa: E402


def _write_list_show_jsonl(path: Path, book_ids: list[int], ratings: list[int] | None = None) -> None:
    ratings = ratings or [0] * len(book_ids)
    record = {
        "book_urls": [f"/book/show/{bid}" for bid in book_ids],
        "titles": [f"Title {bid}" for bid in book_ids],
        "authors": [f"Author {bid}" for bid in book_ids],
        "rating_texts": [f"4.20 avg rating — {r:,} ratings" for r in ratings],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


SEED_YAML = """
lists:
  - list_id: 1
    slug: Anchor
    genre: general
    list_type: anchor
  - list_id: 2
    slug: GenreA
    genre: a
  - list_id: 3
    slug: GenreB
    genre: b
"""


class TestLoadSeedLists(unittest.TestCase):
    def test_defaults_list_type_to_genre(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed.yaml"
            path.write_text(SEED_YAML, encoding="utf-8")
            seed = load_seed_lists(path)
        self.assertEqual(seed[1].list_type, "anchor")
        self.assertEqual(seed[2].list_type, "genre")
        self.assertEqual(seed[3].list_type, "genre")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_seed_lists(Path("/nonexistent/seed.yaml")), {})


class TestLoadCheckpointStatus(unittest.TestCase):
    def test_reads_status_written_by_real_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "checkpoint.sqlite"
            cp = Checkpoint(db_path)
            cp.upsert_seed([SeedList(1, "A", "g"), SeedList(2, "B", "g")])
            cp.record_result(1, status=STATUS_DONE, error=None, output_path="x", book_count=5)
            cp.close()

            status = load_checkpoint_status(db_path)
        self.assertEqual(status, {1: STATUS_DONE, 2: STATUS_PENDING})

    def test_missing_db_returns_empty(self) -> None:
        self.assertEqual(load_checkpoint_status(Path("/nonexistent/checkpoint.sqlite")), {})


class TestSetMath(unittest.TestCase):
    def test_jaccard(self) -> None:
        self.assertAlmostEqual(jaccard({1, 2, 3}, {2, 3, 4}), 2 / 4)
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_containment_is_asymmetric(self) -> None:
        small = {1, 2}
        big = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        self.assertEqual(containment(small, big), 1.0)
        self.assertEqual(containment(big, small), 0.2)
        self.assertEqual(containment(set(), big), 0.0)

    def test_top_n_book_ids(self) -> None:
        ratings = {1: 10, 2: 50, 3: 5, 4: 100}
        self.assertEqual(top_n_book_ids(ratings, 2), {4, 2})


class TestEvaluateLists(unittest.TestCase):
    def _seed(self, **overrides) -> dict:
        base = {
            1: SeedListInfo(1, "Anchor", "general", "anchor"),
            2: SeedListInfo(2, "GenreA", "a", "genre"),
            3: SeedListInfo(3, "GenreB", "b", "genre"),
        }
        base.update(overrides)
        return base

    def test_anchor_always_kept_even_if_it_would_otherwise_be_flagged(self) -> None:
        # Only seed the anchor + one genre list -- list 3 isn't relevant here
        # and would otherwise show up as "not yet evaluated" (no status/data).
        seed = {k: v for k, v in self._seed().items() if k in (1, 2)}
        # Anchor list (1) is wholly contained in list 2 -- would trip the
        # subset rule if it weren't exempt.
        done = {
            1: ListData(1, book_ids={10, 11}, ratings={10: 5, 11: 5}),
            2: ListData(2, book_ids={10, 11, 12, 13, 14}, ratings={10: 5, 11: 5, 12: 5, 13: 5, 14: 5}),
        }
        status = {1: DONE_STATUS, 2: DONE_STATUS}
        recs, not_evaluated = evaluate_lists(seed, status, done, top_n=10)
        by_id = {r.list_id: r for r in recs}
        self.assertEqual(by_id[1].outcome, OUTCOME_KEEP)
        self.assertIn("anchor", by_id[1].reasons[0])
        self.assertEqual(not_evaluated, [])

    def test_subset_rule_fires_on_high_containment(self) -> None:
        seed = self._seed()
        # List 2 (10 books) is 90% contained in list 3, but list 3 is much
        # bigger overall so the reverse containment stays well under 70%.
        list2_books = set(range(100, 110))
        list3_books = set(range(100, 109)) | set(range(2000, 2050))
        done = {
            2: ListData(2, book_ids=list2_books, ratings={b: 1 for b in list2_books}),
            3: ListData(3, book_ids=list3_books, ratings={b: 1 for b in list3_books}),
        }
        status = {2: DONE_STATUS, 3: DONE_STATUS}
        recs, _ = evaluate_lists(seed, status, done, top_n=10)
        by_id = {r.list_id: r for r in recs}
        self.assertEqual(by_id[2].outcome, OUTCOME_DEPRECATE)
        self.assertIn("subset", by_id[2].reasons[0])
        self.assertEqual(by_id[2].metrics["max_containment_other_list_id"], 3)
        # Containment the other direction is much lower -- list 3 must stay.
        self.assertEqual(by_id[3].outcome, OUTCOME_KEEP)

    def test_overlap_alone_does_not_deprecate_without_low_marginal_gain(self) -> None:
        seed = self._seed()
        # High jaccard between 2 and 3 (~0.39), but list 2 still adds 60
        # unique books beyond list 3 -- the compound rule must NOT fire on
        # overlap alone when marginal gain isn't also low. marginal_gain_top_n
        # is set larger than the whole fixture universe so every book counts
        # towards the target, regardless of the (all-tied) rating values.
        shared = set(range(0, 40))
        list2_only = set(range(1000, 1060))  # 60 unique-to-list2 books
        list3_only = {2000, 2001}
        list2_books = shared | list2_only
        list3_books = shared | list3_only
        done = {
            2: ListData(2, book_ids=list2_books, ratings={b: 1 for b in list2_books}),
            3: ListData(3, book_ids=list3_books, ratings={b: 1 for b in list3_books}),
        }
        status = {2: DONE_STATUS, 3: DONE_STATUS}
        recs, _ = evaluate_lists(seed, status, done, top_n=10, marginal_gain_top_n=1000)
        by_id = {r.list_id: r for r in recs}
        self.assertGreater(by_id[2].metrics["max_jaccard"], 0.35, "fixture must actually exercise high overlap")
        self.assertGreaterEqual(by_id[2].metrics["marginal_gain_top1000"], 50)
        self.assertEqual(by_id[2].outcome, OUTCOME_KEEP, "overlap alone must not be sufficient to deprecate")

    def test_overlap_plus_low_marginal_gain_deprecates(self) -> None:
        seed = self._seed()
        # Same shape as the "overlap alone" fixture, but list 2 now only
        # contributes 18 unique books -- below the low-marginal-gain bar --
        # while staying under the subset containment threshold (40/58 ≈ 0.69).
        shared = set(range(0, 40))
        list2_only = set(range(1000, 1018))  # 18 unique-to-list2 books
        list3_only = set(range(2000, 2020))  # keeps list3's own containment < 0.7 too
        list2_books = shared | list2_only
        list3_books = shared | list3_only
        done = {
            2: ListData(2, book_ids=list2_books, ratings={b: 1 for b in list2_books}),
            3: ListData(3, book_ids=list3_books, ratings={b: 1 for b in list3_books}),
        }
        status = {2: DONE_STATUS, 3: DONE_STATUS}
        # Sanity-check the fixture itself doesn't accidentally trip the subset
        # rule instead (containment(2, 3) = 40/58 ≈ 0.69, at or under 0.7).
        self.assertLessEqual(containment(list2_books, list3_books), 0.7)

        recs, _ = evaluate_lists(seed, status, done, top_n=10, marginal_gain_top_n=1000)
        by_id = {r.list_id: r for r in recs}
        self.assertEqual(by_id[2].outcome, OUTCOME_DEPRECATE)
        self.assertIn("overlap", by_id[2].reasons[0])

    def test_lists_not_done_are_excluded_from_evaluation(self) -> None:
        seed = self._seed()
        done = {2: ListData(2, book_ids={1, 2}, ratings={1: 1, 2: 1})}
        status = {2: DONE_STATUS, 3: "pending"}
        recs, not_evaluated = evaluate_lists(seed, status, done, top_n=10)
        rec_ids = {r.list_id for r in recs}
        self.assertEqual(rec_ids, {2})
        self.assertIn(3, not_evaluated)
        # Anchor list 1 has no scrape data/status at all -- also excluded, not crashed on.
        self.assertIn(1, not_evaluated)


class TestGreedySetCover(unittest.TestCase):
    def test_picks_highest_marginal_coverage_first(self) -> None:
        seed = {
            1: SeedListInfo(1, "Big", "g", "genre"),
            2: SeedListInfo(2, "Small", "g", "genre"),
        }
        done = {
            1: ListData(1, book_ids={1, 2, 3, 4}, ratings={}),
            2: ListData(2, book_ids={4, 5}, ratings={}),
        }
        order = greedy_set_cover(done, target={1, 2, 3, 4, 5}, seed=seed)
        self.assertEqual(order[0]["list_id"], 1)
        self.assertEqual(order[0]["new_coverage"], 4)
        self.assertEqual(order[1]["list_id"], 2)
        self.assertEqual(order[1]["new_coverage"], 1)

    def test_stops_when_no_list_adds_new_coverage(self) -> None:
        seed = {1: SeedListInfo(1, "A", "g", "genre")}
        done = {1: ListData(1, book_ids={1, 2}, ratings={})}
        order = greedy_set_cover(done, target={99}, seed=seed)
        self.assertEqual(order, [])


class TestBuildReportEndToEnd(unittest.TestCase):
    def test_report_shape_with_real_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            raw_dir = tmp / "raw"
            raw_dir.mkdir()
            _write_list_show_jsonl(raw_dir / "2.jsonl", [1, 2, 3], [10, 20, 30])
            _write_list_show_jsonl(raw_dir / "3.jsonl", [3, 4, 5], [30, 40, 50])

            seed_dict = {
                2: SeedListInfo(2, "GenreA", "a", "genre"),
                3: SeedListInfo(3, "GenreB", "b", "genre"),
            }
            status = {2: DONE_STATUS, 3: DONE_STATUS}
            done_lists = {
                2: load_list_data(raw_dir, 2),
                3: load_list_data(raw_dir, 3),
            }
            report = build_report(seed_dict, status, done_lists, top_n=5)
        self.assertEqual(report["evaluated"], [2, 3])
        self.assertEqual(report["corpus"]["unique_books"], 5)
        self.assertEqual(len(report["recommendations"]), 2)


class TestApplyDeprecations(unittest.TestCase):
    def test_writes_new_deprecation_entries(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed_list_overrides.yaml"
            recs = [
                Recommendation(2, "GenreA", OUTCOME_DEPRECATE, ["subset: 90% in list 3"], {"size": 10}),
                Recommendation(3, "GenreB", OUTCOME_KEEP, ["no rule matched"], {"size": 20}),
            ]
            added = apply_deprecations(path, recs)
            self.assertEqual(added, 1)
            overrides = load_overrides(path)
            self.assertEqual(set(overrides.keys()), {2})
            self.assertEqual(overrides[2]["curation_status"], "deprecated")
            self.assertIn("set_at", overrides[2])

    def test_does_not_overwrite_existing_manual_entry(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed_list_overrides.yaml"
            path.write_text(
                "overrides:\n"
                "  2:\n"
                "    curation_status: active\n"
                "    reason: manually kept despite rule\n"
                "    set_at: '2020-01-01T00:00:00+00:00'\n",
                encoding="utf-8",
            )
            recs = [Recommendation(2, "GenreA", OUTCOME_DEPRECATE, ["subset: 90% in list 3"], {"size": 10})]
            added = apply_deprecations(path, recs)
            self.assertEqual(added, 0, "an existing entry must never be silently overwritten")
            overrides = load_overrides(path)
            self.assertEqual(overrides[2]["curation_status"], "active")

    def test_returns_zero_when_nothing_to_deprecate(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seed_list_overrides.yaml"
            recs = [Recommendation(2, "GenreA", OUTCOME_KEEP, ["no rule matched"], {"size": 10})]
            added = apply_deprecations(path, recs)
            self.assertEqual(added, 0)
            self.assertFalse(path.exists(), "must not create the file when there's nothing to write")

    def test_load_overrides_missing_file_is_empty(self) -> None:
        self.assertEqual(load_overrides(Path("/nonexistent/overrides.yaml")), {})


if __name__ == "__main__":
    unittest.main()
