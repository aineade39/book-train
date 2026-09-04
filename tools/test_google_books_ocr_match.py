#!/usr/bin/env python3
"""Unit tests for tools/google_books_ocr_match.py (no network).

Run: .venv/bin/python3 tools/test_google_books_ocr_match.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.google_books_ocr_match import (  # noqa: E402
    LookupResult,
    build_query_strategies,
    build_markdown_report,
    compare_run,
    fetch_volumes,
    longest_token_phrase,
    lookup_ocr,
    resolve_google_books_api_key,
    significant_tokens,
    ComparisonRow,
)


class QueryStrategyTests(unittest.TestCase):
    def test_significant_tokens_drop_boilerplate(self) -> None:
        ocr = "lonely planet Spain Pull-out map"
        toks = significant_tokens(ocr)
        self.assertIn("spain", toks)
        self.assertNotIn("lonely", toks)
        self.assertNotIn("planet", toks)
        self.assertNotIn("pull", toks)

    def test_longest_phrase_preserves_ocr_order(self) -> None:
        ocr = "DAY HIKING SNOQUALMIE REGION"
        phrase = longest_token_phrase(ocr)
        self.assertEqual(phrase, "day hiking snoqualmie region")

    def test_build_strategies_includes_tiers(self) -> None:
        ocr = "BACKPACKING WASHINGTON 2nd edition"
        names = [s.name for s in build_query_strategies(ocr)]
        self.assertIn("intitle_phrase", names)
        self.assertIn("intitle_tokens", names)
        self.assertIn("fulltext", names)

    def test_intitle_phrase_uses_quotes(self) -> None:
        ocr = "Mountaineering The Freedom of the Hills"
        strategies = build_query_strategies(ocr)
        phrase = next(s for s in strategies if s.name == "intitle_phrase")
        self.assertTrue(phrase.q.startswith('intitle:"'))
        self.assertIn("freedom", phrase.q)


class LookupTests(unittest.TestCase):
    _MOCK_ITEMS = [
        {
            "id": "vol1",
            "volumeInfo": {
                "title": "Day Hiking Snoqualmie Region",
                "authors": ["Mountaineers Books"],
                "publishedDate": "2014",
            },
        },
        {
            "id": "vol2",
            "volumeInfo": {
                "title": "Go Hiking!",
                "authors": ["Someone"],
                "publishedDate": "2000",
            },
        },
    ]

    def test_lookup_picks_best_fuzzy_match_across_strategies(self) -> None:
        ocr = "DAY HIKING SNOQUALMIE REGION"

        def fake_fetch(q: str, **kwargs):  # noqa: ANN001
            if "snoqualmie" in q.lower():
                return self._MOCK_ITEMS
            return [self._MOCK_ITEMS[1]]

        with patch("tools.google_books_ocr_match.fetch_volumes", side_effect=fake_fetch):
            result = lookup_ocr(ocr, delay_s=0, strategies=build_query_strategies(ocr))

        self.assertIsInstance(result, LookupResult)
        self.assertEqual(result.api_title, "Day Hiking Snoqualmie Region")
        self.assertGreater(result.ocr_api_fuzzy, 80)
        self.assertEqual(result.volume_id, "vol1")

    def test_lookup_empty_ocr(self) -> None:
        result = lookup_ocr("  ", delay_s=0)
        self.assertEqual(result.error, "empty OCR")
        self.assertIsNone(result.api_title)

    def test_fetch_volumes_builds_expected_url(self) -> None:
        captured: dict = {}

        class FakeResp:
            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(self, *args):  # noqa: ANN002, ANN204
                return False

            def read(self) -> bytes:
                return json.dumps({"items": []}).encode()

        def fake_urlopen(req, timeout=0):  # noqa: ANN001
            captured["url"] = req.full_url
            return FakeResp()

        with patch("urllib.request.urlopen", fake_urlopen):
            fetch_volumes('intitle:"day hiking"', api_key="test-key", max_results=5)

        self.assertIn("printType=books", captured["url"])
        self.assertIn("langRestrict=en", captured["url"])
        self.assertIn("key=test-key", captured["url"])


class ApiKeyResolutionTests(unittest.TestCase):
    def test_env_overrides_gcloud(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_BOOKS_API_KEY": "env-key", "GOOGLE_API_KEY": "other"}):
            with patch("tools.google_books_ocr_match.subprocess.run") as mock_run:
                self.assertEqual(resolve_google_books_api_key(), "env-key")
                mock_run.assert_not_called()

    def test_gcloud_fallback_parses_keystring(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_BOOKS_API_KEY", "GOOGLE_API_KEY")}
        fake = unittest.mock.Mock()
        fake.returncode = 0
        fake.stdout = "AIzaSyTestKeyFromGcloud\n"

        with patch.dict(os.environ, env, clear=True):
            with patch("tools.google_books_ocr_match.subprocess.run", return_value=fake) as mock_run:
                self.assertEqual(resolve_google_books_api_key(), "AIzaSyTestKeyFromGcloud")
                argv = mock_run.call_args.args[0]
                self.assertIn("book-train-google-books", argv)
                self.assertTrue(any("intricate-idiom-505902-m5" in a for a in argv))


class ReportTests(unittest.TestCase):
    def test_markdown_report_contains_summary(self) -> None:
        rows = [
            ComparisonRow(
                spine_id="abc12345",
                ocr_text="DAY HIKING SNOQUALMIE REGION",
                oracle_title="Day Hiking Snoqualmie Region",
                oracle_author="Mountaineers",
                api_title="Day Hiking Snoqualmie Region",
                api_authors="Mountaineers Books",
                winning_strategy="intitle_phrase",
                ocr_api_fuzzy=95.0,
                api_oracle_fuzzy=100.0,
                ocr_oracle_fuzzy=100.0,
                volume_id="vol1",
            )
        ]
        md = build_markdown_report(rows, run_json=Path("run.json"), oracle_path=Path("oracle.json"))
        self.assertIn("Google Books API vs spine OCR", md)
        self.assertIn("DAY HIKING SNOQUALMIE REGION", md)
        self.assertIn("Day Hiking Snoqualmie Region", md)


if __name__ == "__main__":
    unittest.main()
