#!/usr/bin/env python3
"""Unit tests for tools/catalog/write_book_show_api_chunked_profile.py
(run: python tools/catalog/test_write_book_show_api_chunked_profile.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.write_book_show_api_chunked_profile import (  # noqa: E402
    BASE_PROFILE_ID,
    CHUNKED_PROFILE_ID,
    DEFAULT_MAX_REQUESTS,
    write_chunked_profile,
)


def _write_base_profile(harness_root: Path, policy: dict | None = None) -> Path:
    profiles_dir = harness_root / "sites" / "goodreads" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": BASE_PROFILE_ID,
        "mode": "api",
        "start_url": "https://www.goodreads.com/book/show/{book_id}",
        "api": {"extract": {"fields": {"legacy_id": "legacy_id"}, "required_fields": ["legacy_id"]}},
        "policy": policy or {"delay_ms_min": 3000, "delay_ms_max": 8000},
    }
    path = profiles_dir / f"{BASE_PROFILE_ID}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class TestWriteChunkedProfile(unittest.TestCase):
    def test_default_max_requests_is_120(self) -> None:
        self.assertEqual(DEFAULT_MAX_REQUESTS, 120)

    def test_copies_base_profile_with_max_requests_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_root = Path(tmp)
            _write_base_profile(harness_root)

            out_path = write_chunked_profile(harness_root, max_requests=200)

            self.assertTrue(out_path.exists())
            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["id"], CHUNKED_PROFILE_ID)
            self.assertEqual(data["policy"]["max_requests"], 200)
            self.assertEqual(data["policy"]["delay_ms_min"], 3000)

    def test_preserves_api_extract_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_root = Path(tmp)
            _write_base_profile(harness_root)

            out_path = write_chunked_profile(harness_root, max_requests=200)

            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["api"]["extract"]["required_fields"], ["legacy_id"])

    def test_rewrite_reflects_updated_base_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_root = Path(tmp)
            _write_base_profile(harness_root, policy={"delay_ms_min": 3000, "delay_ms_max": 8000})
            write_chunked_profile(harness_root, max_requests=200)

            _write_base_profile(harness_root, policy={"delay_ms_min": 1000, "delay_ms_max": 2000})
            out_path = write_chunked_profile(harness_root, max_requests=200)

            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["policy"]["delay_ms_min"], 1000)

    def test_delay_overrides_replace_base_profile_delays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_root = Path(tmp)
            _write_base_profile(harness_root, policy={"delay_ms_min": 3000, "delay_ms_max": 8000})

            out_path = write_chunked_profile(harness_root, max_requests=200, delay_ms_min=1500, delay_ms_max=4000)

            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["policy"]["delay_ms_min"], 1500)
            self.assertEqual(data["policy"]["delay_ms_max"], 4000)

    def test_omitted_delay_overrides_inherit_base_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness_root = Path(tmp)
            _write_base_profile(harness_root, policy={"delay_ms_min": 3000, "delay_ms_max": 8000})

            out_path = write_chunked_profile(harness_root, max_requests=200)

            data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["policy"]["delay_ms_min"], 3000)
            self.assertEqual(data["policy"]["delay_ms_max"], 8000)

    def test_missing_base_profile_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                write_chunked_profile(Path(tmp), max_requests=200)


if __name__ == "__main__":
    unittest.main()
