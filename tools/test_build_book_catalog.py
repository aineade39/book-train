#!/usr/bin/env python3
"""Unit tests for tools/build_book_catalog.py's custom_build_script guard
(run: python tools/test_build_book_catalog.py).

Only exercises the guard that keeps profiles like `ios_en_shelf` (built by
tools/catalog/build_ios_en_from_goodreads.py, which needs SQL mutation this
generic subset-from path can't do) from ever being built through here —
does not exercise the real OL download/build pipeline.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import tools.build_book_catalog as build_book_catalog  # noqa: E402


class TestCustomBuildScriptGuard(unittest.TestCase):
    def test_explicit_profile_with_custom_build_script_is_rejected(self) -> None:
        with patch("sys.argv", ["build_book_catalog.py", "--profile", "ios_en_shelf"]):
            with patch("tools.build_book_catalog.run") as mock_run:
                rc = build_book_catalog.main()
        self.assertEqual(rc, 1)
        mock_run.assert_not_called()

    def test_all_excludes_custom_build_script_profiles_without_erroring(self) -> None:
        profiles = build_book_catalog.load_profiles(build_book_catalog.PROFILES_PATH)
        self.assertIn("ios_en_shelf", profiles, "profiles.json should already declare ios_en_shelf")
        self.assertEqual(profiles["ios_en_shelf"].get("custom_build_script"), "tools/catalog/build_ios_en_from_goodreads.py")

        with patch("sys.argv", ["build_book_catalog.py", "--all", "--skip-download", "--reuse-intermediate"]):
            with patch("tools.build_book_catalog.intermediate_fresh", return_value=True):
                with patch("tools.build_book_catalog.build_profile") as mock_build_profile:
                    mock_build_profile.return_value = Path("/tmp/fake.sqlite")
                    build_book_catalog.main()
        built_names = {call.args[0] for call in mock_build_profile.call_args_list}
        self.assertNotIn("ios_en_shelf", built_names)


if __name__ == "__main__":
    unittest.main()
