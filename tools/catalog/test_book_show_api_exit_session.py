#!/usr/bin/env python3
"""Unit tests for tools/catalog/book_show_api_exit_session.py
(run: python tools/catalog/test_book_show_api_exit_session.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.book_show_api_exit_session import (  # noqa: E402
    ACTION_CYCLE_SLEEP,
    ACTION_HARD_STOP,
    ACTION_ROTATE,
    CHUNKED_PROFILE_ID,
    DISCOVER_BOOK_HREF,
    DISCOVER_LIST_URL,
    DISCOVER_URL,
    HOME_URL,
    PREPARE_DISPLAY_FAIL,
    PREPARE_FAIL,
    PREPARE_HARD_STOP,
    PREPARE_OK,
    DiscoverFailState,
    apply_fail,
    discover_command,
    is_display_failure,
    load_fail_state,
    main,
    next_fail_action,
    prepare_exit,
    prepare_exit_result,
    rotate_limit,
    save_fail_state,
    wipe_chunked_profile,
)


class TestDiscoverCommand(unittest.TestCase):
    def test_shares_chunked_profile_and_human_warmup(self) -> None:
        root = Path("/tmp/scrape-harness")
        cmd = discover_command(root)
        self.assertIn(CHUNKED_PROFILE_ID, cmd)
        self.assertIn(HOME_URL, cmd)
        self.assertIn(DISCOVER_LIST_URL, cmd)
        self.assertIn(DISCOVER_URL, cmd)
        self.assertIn(DISCOVER_BOOK_HREF, cmd)
        self.assertIn("--fresh-browser", cmd)
        self.assertNotIn("--keep-profile", cmd)
        self.assertEqual(cmd.count("--warmup-url"), 2)
        home_at = cmd.index(HOME_URL)
        list_at = cmd.index(DISCOVER_LIST_URL)
        self.assertEqual(cmd[home_at - 1], "--warmup-url")
        self.assertEqual(cmd[list_at - 1], "--warmup-url")
        self.assertLess(home_at, list_at)

    def test_keep_profile_flag(self) -> None:
        cmd = discover_command(Path("/tmp/h"), fresh_browser=False)
        self.assertIn("--keep-profile", cmd)
        self.assertNotIn("--fresh-browser", cmd)


class TestFailSafeguard(unittest.TestCase):
    def test_rotate_until_city_pool_exhausted(self) -> None:
        self.assertEqual(next_fail_action(1, 0, limit=8), ACTION_ROTATE)
        self.assertEqual(next_fail_action(7, 0, limit=8), ACTION_ROTATE)
        self.assertEqual(next_fail_action(8, 0, limit=8), ACTION_CYCLE_SLEEP)

    def test_second_full_cycle_hard_stops(self) -> None:
        self.assertEqual(next_fail_action(8, 1, limit=8, max_cycles=2), ACTION_HARD_STOP)

    def test_apply_fail_resets_consecutive_on_cycle_sleep(self) -> None:
        state, action = apply_fail(DiscoverFailState(consecutive=7, cycles=0), limit=8)
        self.assertEqual(action, ACTION_CYCLE_SLEEP)
        self.assertEqual(state, DiscoverFailState(consecutive=0, cycles=1))

    def test_rotate_limit_follows_city_pool(self) -> None:
        self.assertEqual(rotate_limit(("nyc", "lax", "chi")), 3)

    def test_cli_records_action_and_hard_stop_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "discover_fail_state.json"
            rc = main(["next-fail-action", "--state-path", str(state_path), "--consecutive", "7", "--limit", "8", "--cycles", "1", "--max-cycles", "2"])
            self.assertEqual(rc, PREPARE_HARD_STOP)
            saved = load_fail_state(state_path)
            self.assertEqual(saved.consecutive, 8)
            self.assertEqual(saved.cycles, 2)


class TestPrepareExit(unittest.TestCase):
    def test_wipes_chunked_profile_then_invokes_discover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / ".data" / "browser_profiles" / "goodreads" / "book_show_api_chunked"
            profile.mkdir(parents=True)
            (profile / "Cookies").write_text("old", encoding="utf-8")
            harness_bin = root / ".venv" / "bin" / "scrape-harness"
            harness_bin.parent.mkdir(parents=True)
            harness_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            harness_bin.chmod(0o755)
            seen: list[list[str]] = []

            def run_fn(argv: list[str]) -> SimpleNamespace:
                seen.append(argv)
                return SimpleNamespace(returncode=0)

            self.assertTrue(prepare_exit(root, run_fn=run_fn))
            self.assertFalse(profile.exists())
            self.assertEqual(len(seen), 1)
            self.assertIn("--fresh-browser", seen[0])
            self.assertIn(HOME_URL, seen[0])

    def test_missing_harness_binary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(prepare_exit(Path(tmp), run_fn=lambda _argv: SimpleNamespace(returncode=0)))

    def test_wipe_clears_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / ".data" / "locks" / "goodreads.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("held", encoding="utf-8")
            wipe_chunked_profile(root)
            self.assertFalse(lock.exists())

    def test_prepare_cli_missing_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["prepare-exit", "--harness-root", tmp]), PREPARE_FAIL)

    def test_display_failure_exits_distinct_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness_bin = root / ".venv" / "bin" / "scrape-harness"
            harness_bin.parent.mkdir(parents=True)
            harness_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            harness_bin.chmod(0o755)

            def run_fn(_argv: list[str]) -> SimpleNamespace:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "Looks like you launched a headed browser without having a XServer running.\n"
                        "Missing X server or $DISPLAY\n"
                        "ozone_platform_x11.cc:257\n"
                    ),
                )

            result = prepare_exit_result(root, run_fn=run_fn)
            self.assertFalse(result.ok)
            self.assertTrue(result.display_failure)
            with patch(
                "tools.catalog.book_show_api_exit_session.prepare_exit_result",
                return_value=result,
            ):
                self.assertEqual(main(["prepare-exit", "--harness-root", str(root)]), PREPARE_DISPLAY_FAIL)


class TestIsDisplayFailure(unittest.TestCase):
    def test_playwright_banner(self) -> None:
        text = (
            "Looks like you launched a headed browser without having a XServer running.\n"
            "Set either 'headless: true' or use 'xvfb-run'\n"
        )
        self.assertTrue(is_display_failure(text))

    def test_chrome_missing_x(self) -> None:
        self.assertTrue(is_display_failure("Missing X server or $DISPLAY"))

    def test_ozone_marker(self) -> None:
        self.assertTrue(is_display_failure("ERROR:ui/ozone/platform/x11/ozone_platform_x11.cc:257"))

    def test_ordinary_discover_miss_is_not_display(self) -> None:
        self.assertFalse(is_display_failure("Discovery failed: No <script id='__NEXT_DATA__'> tag found"))


class TestFailStateRoundtrip(unittest.TestCase):
    def test_save_load_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "discover_fail_state.json"
            save_fail_state(path, DiscoverFailState(consecutive=3, cycles=1))
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["consecutive"], 3)
            self.assertEqual(main(["reset-fail", "--state-path", str(path)]), PREPARE_OK)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
