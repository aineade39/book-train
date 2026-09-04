#!/usr/bin/env python3
"""Unit tests for tools/catalog/container_scrape.py
(run: python tools/catalog/test_container_scrape.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.container_scrape import (  # noqa: E402
    backup_verified,
    compose_bind_matches_host,
    jsonl_stat,
    write_marker,
)


class TestJsonlStat(unittest.TestCase):
    def test_requires_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                jsonl_stat(path)

    def test_counts_lines_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book_show_api.jsonl"
            path.write_text("a\nb\n", encoding="utf-8")
            stat = jsonl_stat(path)
            self.assertEqual(stat["lines"], 2)
            self.assertEqual(stat["bytes"], 4)


class TestComposeBind(unittest.TestCase):
    def test_matches_resolved_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "book-spines"
            root.mkdir()
            with patch.dict("os.environ", {"BOOK_SPINES_DATA": str(root)}, clear=False):
                self.assertTrue(compose_bind_matches_host())


class TestBackupVerified(unittest.TestCase):
    def test_fails_when_nas_missing(self) -> None:
        with patch("tools.catalog.container_scrape.NAS_ROOT", Path("/no/such/nas")):
            ok, detail = backup_verified()
            self.assertFalse(ok)
            self.assertIn("NAS", detail)


class TestMarker(unittest.TestCase):
    def test_write_roundtrip_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "container_migration.json"
            with patch("tools.catalog.container_scrape.marker_path", return_value=path):
                with patch("tools.catalog.container_scrape.host_data_root", return_value=Path(tmp)):
                    with patch(
                        "tools.catalog.container_scrape.jsonl_path",
                        return_value=Path(tmp) / "book_show_api.jsonl",
                    ):
                        write_marker(run_id="2026-08-31T000000Z", stat={"bytes": 10, "lines": 2})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["run_id"], "2026-08-31T000000Z")
            self.assertEqual(data["bytes"], 10)
            self.assertEqual(data["lines"], 2)
            self.assertIn("date", data)


if __name__ == "__main__":
    unittest.main()
