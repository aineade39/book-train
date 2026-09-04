#!/usr/bin/env python3
"""Unit tests for tools/catalog/book_show_api_session_health.py"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.book_show_api_session_health import (  # noqa: E402
    RECOVER_ESCALATED,
    RECOVER_HARD_STOP,
    RECOVER_OK,
    RecoveryState,
    SpikeKind,
    analyze_chunk,
    classify_chunk,
    diagnose_catalog,
    escalate_controlled_stop,
    is_failure_spike,
    load_recovery_state,
    main,
    pick_probe_book_id,
    popularity_sidecar_is_corrupt,
    post_rotate_cooldown,
    queue_ordering_suspect,
    recover_from_spike,
    repair_catalog,
    reset_recovery_state,
    save_recovery_state,
)
from tools.catalog.extract_remaining_ids import (  # noqa: E402
    update_retry_after,
    write_popularity_sidecar,
)
from tools.catalog.list_show_popularity import BookPopularity  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _empty_catalog(tmp: Path) -> Path:
    catalog = tmp / "catalog"
    catalog.mkdir()
    return catalog


class TestAnalyzeChunk(unittest.TestCase):
    def test_counts_successes_and_warnings_since_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            _write_jsonl(
                path,
                [
                    {"legacy_id": 1, "isbn13": "9780000000001"},
                    {"_scrape_warning": "incomplete_record"},
                    {"legacy_id": 2, "isbn13": "9780000000002"},
                    {"_scrape_warning": "incomplete_record"},
                ],
            )
            stats = analyze_chunk(path, since_line=2)
            self.assertEqual(stats.total, 2)
            self.assertEqual(stats.successes, 1)
            self.assertEqual(stats.warnings, 1)
            self.assertEqual(stats.warning_types["incomplete_record"], 1)
            self.assertEqual(stats.trailing_warnings, 1)

    def test_failure_spike_requires_minimum_sample(self) -> None:
        from tools.catalog.book_show_api_session_health import ChunkStats

        small = ChunkStats(total=5, successes=0, warnings=5, trailing_warnings=5)
        self.assertFalse(is_failure_spike(small, min_records=20, trailing_abort=8))
        large = ChunkStats(total=50, successes=5, warnings=45)
        self.assertTrue(is_failure_spike(large, min_records=20, threshold=0.80))

    def test_failure_spike_from_trailing_warn_run(self) -> None:
        from tools.catalog.book_show_api_session_health import ChunkStats

        mixed = ChunkStats(total=108, successes=100, warnings=8, trailing_warnings=8)
        self.assertTrue(is_failure_spike(mixed, min_records=20, trailing_abort=8))
        short_tail = ChunkStats(total=106, successes=100, warnings=6, trailing_warnings=6)
        self.assertFalse(is_failure_spike(short_tail, min_records=20, trailing_abort=8))


class TestQueueOrderingSuspect(unittest.TestCase):
    def test_detects_empty_popularity_head_with_rated_remaining(self) -> None:
        popularity = {1: BookPopularity(ratings_count=0), 99: BookPopularity(ratings_count=500_000)}
        remaining = {1, 99}
        ordered = [1, 99]
        self.assertTrue(queue_ordering_suspect(ordered, popularity, remaining, top_n=1, ratings_fraction=0.1))

    def test_ok_when_head_has_ratings(self) -> None:
        popularity = {99: BookPopularity(ratings_count=500_000), 1: BookPopularity(ratings_count=0)}
        remaining = {1, 99}
        ordered = [99, 1]
        self.assertFalse(queue_ordering_suspect(ordered, popularity, remaining))


class TestDiagnoseCatalogQueue(unittest.TestCase):
    def test_diagnose_honors_cooling_ids_and_cached_popularity(self) -> None:
        """10 unrated never-tried + 2 high-rated cooling retries.

        If diagnose forgot cooling, remaining would include the rated
        retries (2/12 >= 10%) with an unrated head — ordering suspect.
        Cached sidecar (not raw) is what supplies those retry ratings.
        """
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            raw_dir = catalog / "raw"
            now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
            never_tried = list(range(200, 210))
            cooling = (1, 2)
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": [f"/book/show/{i}.Pad" for i in never_tried]
                        + [f"/book/show/{i}.Cool" for i in cooling],
                        "rating_texts": ["4.0 avg rating — 0 ratings"] * 12,
                        "score_texts": [""] * 12,
                        "vote_texts": [""] * 12,
                    }
                ],
            )
            _write_jsonl(
                catalog / "book_show_api.jsonl",
                [
                    {
                        "_scrape_warning": "incomplete_record",
                        "_url": f"https://www.goodreads.com/_next/data/abc/book/show/{i}.json",
                    }
                    for i in cooling
                ],
            )
            update_retry_after({1: 1, 2: 1}, catalog / "book_show_api_retry_after.json", 24.0, now=now)
            write_popularity_sidecar(
                catalog / "book_popularity.json",
                {
                    **{i: BookPopularity(ratings_count=0) for i in never_tried},
                    1: BookPopularity(ratings_count=500_000),
                    2: BookPopularity(ratings_count=400_000),
                },
            )
            retry_before = (catalog / "book_show_api_retry_after.json").read_text(encoding="utf-8")
            pop_before = (catalog / "book_popularity.json").read_text(encoding="utf-8")

            diagnosis = diagnose_catalog(catalog)
            self.assertFalse(diagnosis.issues)
            self.assertEqual((catalog / "book_show_api_retry_after.json").read_text(encoding="utf-8"), retry_before)
            self.assertEqual((catalog / "book_popularity.json").read_text(encoding="utf-8"), pop_before)

    def test_diagnose_does_not_rebuild_empty_popularity_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            raw_dir = catalog / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR", "/book/show/99.Dune"],
                        "rating_texts": ["4.5 avg rating — 100 ratings", "4.6 avg rating — 500 ratings"],
                        "score_texts": ["", ""],
                        "vote_texts": ["", ""],
                    }
                ],
            )
            sidecar = catalog / "book_popularity.json"
            sidecar.write_text('{"books": {}}', encoding="utf-8")
            diagnosis = diagnose_catalog(catalog)
            self.assertTrue(diagnosis.issues)
            self.assertEqual(json.loads(sidecar.read_text())["books"], {})


class TestRepairCatalog(unittest.TestCase):
    def test_repair_rebuilds_empty_popularity_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            raw_dir = catalog / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR", "/book/show/99.Dune"],
                        "rating_texts": ["4.5 avg rating — 100 ratings", "4.6 avg rating — 500 ratings"],
                        "score_texts": ["", ""],
                        "vote_texts": ["", ""],
                    }
                ],
            )
            sidecar = catalog / "book_popularity.json"
            sidecar.write_text('{"books": {}}', encoding="utf-8")
            time.sleep(0.01)

            self.assertTrue(popularity_sidecar_is_corrupt(sidecar, raw_dir))
            diagnosis = diagnose_catalog(catalog)
            self.assertTrue(diagnosis.issues)

            repaired, _ = repair_catalog(catalog)
            self.assertTrue(repaired)
            self.assertGreater(len(json.loads(sidecar.read_text())["books"]), 0)

            ids_remaining = (catalog / "ids_remaining.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(ids_remaining[0], "99")


class TestClassifyChunk(unittest.TestCase):
    def test_hard_block_from_majority_blocked_suspected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"_scrape_warning": "blocked_suspected"}] * 6
                + [{"_scrape_warning": "incomplete_record"}] * 2
                + [{"legacy_id": 1, "isbn13": "9780000000001"}] * 2,
            )
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.HARD_BLOCK)

    def test_catalog_when_popularity_sidecar_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog"
            raw_dir = catalog / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR"],
                        "rating_texts": ["4.5 avg rating — 100 ratings"],
                        "score_texts": [""],
                        "vote_texts": [""],
                    }
                ],
            )
            (catalog / "book_popularity.json").write_text('{"books": {}}', encoding="utf-8")
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "incomplete_record"}] * 20)
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.CATALOG)

    def test_soft_block_when_catalog_ok_and_incomplete_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"_scrape_warning": "incomplete_record"}] * 15
                + [{"_scrape_warning": "json_parse_error"}] * 5,
            )
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.SOFT_BLOCK)

    def test_stale_build_from_majority_json_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"_scrape_warning": "json_parse_error", "_status": 404}] * 20,
            )
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.STALE_BUILD)

    def test_stale_build_from_majority_404_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"_scrape_warning": "incomplete_record", "_status": 404}] * 20,
            )
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.STALE_BUILD)

    def test_stale_build_from_trailing_404s_despite_healthy_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"legacy_id": 1, "isbn13": "9780000000001"}] * 20
                + [{"_scrape_warning": "json_parse_error", "_status": 404}] * 8,
            )
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.STALE_BUILD)
            stats = analyze_chunk(jsonl, 0)
            self.assertTrue(is_failure_spike(stats))

    def test_hard_block_wins_over_catalog_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog"
            raw_dir = catalog / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR"],
                        "rating_texts": ["4.5 avg rating — 100 ratings"],
                        "score_texts": [""],
                        "vote_texts": [""],
                    }
                ],
            )
            (catalog / "book_popularity.json").write_text('{"books": {}}', encoding="utf-8")
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "blocked_suspected"}] * 20)
            self.assertEqual(classify_chunk(jsonl, 0, catalog_dir=catalog), SpikeKind.HARD_BLOCK)


class TestPickProbeBookId(unittest.TestCase):
    def test_uses_last_successful_legacy_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [
                    {"legacy_id": 33, "isbn13": "9780618640157"},
                    {"_scrape_warning": "incomplete_record"},
                    {"legacy_id": 13335037, "isbn13": "9780062024039"},
                ],
            )
            self.assertEqual(pick_probe_book_id(jsonl, Path(tmp)), 13335037)

    def test_falls_back_to_fetched_ids_then_33(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp)
            jsonl = catalog / "missing.jsonl"
            (catalog / "fetched_ids.txt").write_text("99\n100\n", encoding="utf-8")
            self.assertEqual(pick_probe_book_id(jsonl, catalog), 99)
            (catalog / "fetched_ids.txt").unlink()
            self.assertEqual(pick_probe_book_id(jsonl, catalog), 33)


class TestRecoverFromSpike(unittest.TestCase):
    def test_hard_block_returns_hard_stop_without_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "blocked_suspected"}] * 20)
            probed: list[int] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                skip_sleep=True,
                probe_fn=lambda *_a: probed.append(1) or True,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_HARD_STOP)
            self.assertEqual(probed, [])

    def test_soft_block_probe_success_resets_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"legacy_id": 33, "isbn13": "9780618640157"}]
                + [{"_scrape_warning": "incomplete_record"}] * 20,
            )
            state_path = catalog / "spike_recovery_state.json"
            wipes: list[Path] = []
            discovers: list[Path] = []
            sleeps: list[float] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                state_path=state_path,
                skip_sleep=False,
                sleep_fn=sleeps.append,
                wipe_fn=wipes.append,
                discover_fn=lambda root: discovers.append(root) or True,
                probe_fn=lambda *_a: True,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_OK)
            self.assertEqual(len(wipes), 1)
            self.assertEqual(len(discovers), 1)
            self.assertEqual(sleeps, [900])
            self.assertFalse(state_path.exists())

    def test_soft_block_probe_failures_escalate_then_hard_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "incomplete_record"}] * 20)
            state_path = catalog / "spike_recovery_state.json"
            sleeps: list[float] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                state_path=state_path,
                max_tiers=3,
                cooldown_tiers=(10, 20, 30),
                skip_sleep=False,
                sleep_fn=sleeps.append,
                wipe_fn=lambda _root: None,
                discover_fn=lambda _root: True,
                probe_fn=lambda *_a: False,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_HARD_STOP)
            self.assertEqual(sleeps, [10, 20, 30])
            state = load_recovery_state(state_path)
            self.assertEqual(state.tier, 3)
            self.assertEqual(state.consecutive_spikes, 1)

    def test_catalog_repair_path_skips_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog"
            raw_dir = catalog / "raw"
            _write_jsonl(
                raw_dir / "1.jsonl",
                [
                    {
                        "book_urls": ["/book/show/33.LOTR"],
                        "rating_texts": ["4.5 avg rating — 100 ratings"],
                        "score_texts": [""],
                        "vote_texts": [""],
                    }
                ],
            )
            (catalog / "book_popularity.json").write_text('{"books": {}}', encoding="utf-8")
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "incomplete_record"}] * 20)
            probed: list[int] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                skip_sleep=True,
                probe_fn=lambda *_a: probed.append(1) or True,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_OK)
            self.assertEqual(probed, [])
            self.assertGreater(len(json.loads((catalog / "book_popularity.json").read_text())["books"]), 0)

    def test_stale_build_recovers_without_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"legacy_id": 33, "isbn13": "9780618640157"}]
                + [{"_scrape_warning": "json_parse_error", "_status": 404}] * 20,
            )
            state_path = catalog / "spike_recovery_state.json"
            save_recovery_state(state_path, RecoveryState(tier=1, consecutive_spikes=1))
            wipes: list[Path] = []
            discovers: list[Path] = []
            sleeps: list[float] = []
            probed: list[int] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                state_path=state_path,
                skip_sleep=False,
                sleep_fn=sleeps.append,
                wipe_fn=wipes.append,
                discover_fn=lambda root: discovers.append(root) or True,
                probe_fn=lambda _root, book_id, _out: probed.append(book_id) or True,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_OK)
            self.assertEqual(len(discovers), 1)
            self.assertEqual(probed, [33])
            self.assertEqual(wipes, [])
            self.assertEqual(sleeps, [])
            # Successful stale-build recovery must not bump the soft-block tier.
            leftover = load_recovery_state(state_path)
            self.assertEqual(leftover.tier, 1)
            self.assertEqual(leftover.consecutive_spikes, 1)

    def test_stale_build_probe_fail_falls_through_to_soft_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "json_parse_error", "_status": 404}] * 20)
            state_path = catalog / "spike_recovery_state.json"
            discovers: list[Path] = []
            wipes: list[Path] = []
            sleeps: list[float] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                state_path=state_path,
                max_tiers=1,
                cooldown_tiers=(10,),
                stale_retry_cooldowns=(5, 7),
                skip_sleep=False,
                sleep_fn=sleeps.append,
                wipe_fn=wipes.append,
                discover_fn=lambda root: discovers.append(root) or True,
                probe_fn=lambda *_a: False,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_HARD_STOP)
            self.assertEqual(len(discovers), 4)  # immediate + 2 stale retries + soft-block tier
            self.assertEqual(len(wipes), 1)
            self.assertEqual(sleeps, [5, 7, 10])

    def test_stale_build_retry_probe_success_skips_soft_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "json_parse_error", "_status": 404}] * 20)
            state_path = catalog / "spike_recovery_state.json"
            discovers: list[Path] = []
            wipes: list[Path] = []
            sleeps: list[float] = []
            probes = {"n": 0}

            def probe_fn(*_a: object) -> bool:
                probes["n"] += 1
                return probes["n"] >= 2

            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                state_path=state_path,
                stale_retry_cooldowns=(5, 7),
                skip_sleep=False,
                sleep_fn=sleeps.append,
                wipe_fn=wipes.append,
                discover_fn=lambda root: discovers.append(root) or True,
                probe_fn=probe_fn,
                rotate_fn=lambda: False,
            )
            self.assertEqual(rc, RECOVER_OK)
            self.assertEqual(len(discovers), 2)
            self.assertEqual(probes["n"], 2)
            self.assertEqual(wipes, [])
            self.assertEqual(sleeps, [5])
            self.assertFalse(state_path.exists())

    def test_hard_block_rotate_success_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(jsonl, [{"_scrape_warning": "blocked_suspected"}] * 20)
            probed: list[int] = []
            rotated: list[int] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                skip_sleep=True,
                probe_fn=lambda *_a: probed.append(1) or True,
                rotate_fn=lambda: rotated.append(1) or True,
            )
            self.assertEqual(rc, RECOVER_OK)
            self.assertEqual(rotated, [1])
            self.assertEqual(probed, [])

    def test_soft_block_rotate_caps_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            jsonl = Path(tmp) / "out.jsonl"
            _write_jsonl(
                jsonl,
                [{"legacy_id": 33, "isbn13": "9780618640157"}]
                + [{"_scrape_warning": "incomplete_record"}] * 20,
            )
            sleeps: list[float] = []
            rc = recover_from_spike(
                jsonl,
                0,
                harness_root=Path(tmp) / "harness",
                catalog_dir=catalog,
                skip_sleep=False,
                sleep_fn=sleeps.append,
                wipe_fn=lambda _root: None,
                discover_fn=lambda _root: True,
                probe_fn=lambda *_a: True,
                rotate_fn=lambda: True,
            )
            self.assertEqual(rc, RECOVER_OK)
            self.assertEqual(sleeps, [post_rotate_cooldown(900)])
            self.assertEqual(sleeps, [180])

    def test_refresh_build_cli_missing_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["refresh-build", "--harness-root", tmp]), 1)

    def test_reset_state_cli_deletes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            state_path = catalog / "spike_recovery_state.json"
            save_recovery_state(state_path, RecoveryState(tier=2, consecutive_spikes=3))
            self.assertEqual(main(["reset-state", "--catalog-dir", str(catalog)]), 0)
            self.assertFalse(state_path.exists())
            reset_recovery_state(state_path)  # idempotent
            self.assertFalse(state_path.exists())

    def test_escalate_controlled_stop_sleeps_tier_without_rotate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "spike_recovery_state.json"
            sleeps: list[float] = []
            rc = escalate_controlled_stop(
                state_path=state_path,
                sleep_fn=sleeps.append,
            )
            self.assertEqual(rc, RECOVER_ESCALATED)
            self.assertEqual(sleeps, [900])
            state = load_recovery_state(state_path)
            self.assertEqual(state.tier, 1)
            self.assertEqual(state.consecutive_spikes, 1)

    def test_escalate_controlled_stop_hard_stops_after_max_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "spike_recovery_state.json"
            save_recovery_state(state_path, RecoveryState(tier=3, consecutive_spikes=3))
            sleeps: list[float] = []
            rc = escalate_controlled_stop(
                state_path=state_path,
                max_tiers=3,
                sleep_fn=sleeps.append,
            )
            self.assertEqual(rc, RECOVER_HARD_STOP)
            self.assertEqual(sleeps, [])

    def test_escalate_cooldown_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _empty_catalog(Path(tmp))
            rc = main(["escalate-cooldown", "--catalog-dir", str(catalog), "--skip-sleep"])
            self.assertEqual(rc, RECOVER_ESCALATED)


if __name__ == "__main__":
    unittest.main()
