#!/usr/bin/env python3
"""Unit tests for tools/catalog/mullvad_rotate.py
(run: python tools/catalog/test_mullvad_rotate.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.mullvad_rotate import (  # noqa: E402
    DEFAULT_CITIES,
    MullvadStatus,
    maybe_rotate_exit,
    next_city,
    parse_cities,
    parse_status,
    rotate_enabled,
    rotate_exit,
    rotate_exit_gluetun,
)


def _ok(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestCityPool(unittest.TestCase):
    def test_default_eight_us_cities(self) -> None:
        self.assertEqual(DEFAULT_CITIES, ("nyc", "lax", "chi", "dal", "sea", "atl", "mia", "qas"))
        self.assertEqual(parse_cities(""), DEFAULT_CITIES)

    def test_parse_cities_from_env_style_string(self) -> None:
        self.assertEqual(parse_cities("nyc, LAX chi"), ("nyc", "lax", "chi"))

    def test_next_city_skips_current_and_wraps(self) -> None:
        cities = ("nyc", "lax", "chi")
        self.assertEqual(next_city(cities, None), "nyc")
        self.assertEqual(next_city(cities, "sea"), "nyc")
        self.assertEqual(next_city(cities, "nyc"), "lax")
        self.assertEqual(next_city(cities, "chi"), "nyc")

    def test_rotate_enabled_truthy_values(self) -> None:
        self.assertFalse(rotate_enabled({}))
        self.assertTrue(rotate_enabled({"MULLVAD_ROTATE": "1"}))
        self.assertTrue(rotate_enabled({"MULLVAD_ROTATE": "yes"}))
        self.assertFalse(rotate_enabled({"MULLVAD_ROTATE": "0"}))


class TestParseStatus(unittest.TestCase):
    def test_json_hostname(self) -> None:
        raw = json.dumps(
            {
                "state": "connected",
                "details": {"location": {"hostname": "us-nyc-wg-301"}},
            }
        )
        status = parse_status(raw)
        self.assertTrue(status.connected)
        self.assertEqual(status.city, "nyc")
        self.assertEqual(status.hostname, "us-nyc-wg-301")

    def test_verbose_text(self) -> None:
        status = parse_status("Connected\n    Relay:                  us-lax-wg-101\n")
        self.assertTrue(status.connected)
        self.assertEqual(status.city, "lax")

    def test_disconnected_text(self) -> None:
        status = parse_status("Disconnected")
        self.assertFalse(status.connected)
        self.assertIsNone(status.city)


class TestRotateExit(unittest.TestCase):
    def test_dry_run_logs_without_calling_connect(self) -> None:
        calls: list[list[str]] = []

        def run_fn(argv: list[str]) -> subprocess.CompletedProcess:
            calls.append(argv)
            if argv[1] == "status":
                return _ok("Connected\n    Relay: us-nyc-wg-001\n")
            return _ok()

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rotations.jsonl"
            ok = rotate_exit(
                cities=("nyc", "lax"),
                log_path=log,
                dry_run=True,
                run_fn=run_fn,
                bin_path="/usr/bin/mullvad",
            )
            self.assertTrue(ok)
            self.assertTrue(all(c[1] == "status" for c in calls))
            record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["from"], "nyc")
            self.assertEqual(record["to"], "lax")
            self.assertTrue(record["dry_run"])

    def test_connect_sequence_and_confirm(self) -> None:
        calls: list[list[str]] = []
        connected = {"city": "nyc"}

        def run_fn(argv: list[str]) -> subprocess.CompletedProcess:
            calls.append(argv[1:])
            if argv[1] == "status":
                return _ok(f"Connected\n    Relay: us-{connected['city']}-wg-001\n")
            if argv[1] == "disconnect":
                return _ok()
            if argv[1:4] == ["relay", "set", "location"]:
                connected["city"] = argv[5]
                return _ok()
            if argv[1] == "connect":
                return _ok()
            return _ok("", returncode=1)

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rotations.jsonl"
            ok = rotate_exit(
                cities=("nyc", "lax"),
                log_path=log,
                run_fn=run_fn,
                bin_path="/usr/bin/mullvad",
                fetch_ip_fn=lambda: None,
            )
            self.assertTrue(ok)
            self.assertEqual(
                [c[:3] for c in calls if c[0] != "status"],
                [["disconnect"], ["relay", "set", "location"], ["connect", "--wait"]],
            )
            record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["to"], "lax")
            self.assertTrue(record["ok"])

    def test_command_failure_returns_false(self) -> None:
        def run_fn(argv: list[str]) -> subprocess.CompletedProcess:
            if argv[1] == "status":
                return _ok("Disconnected")
            if argv[1] == "disconnect":
                return _ok("", returncode=1)
            return _ok()

        with tempfile.TemporaryDirectory() as tmp:
            ok = rotate_exit(
                cities=("nyc", "lax"),
                log_path=Path(tmp) / "rotations.jsonl",
                run_fn=run_fn,
                bin_path="/usr/bin/mullvad",
            )
            self.assertFalse(ok)

    def test_maybe_rotate_skips_when_disabled(self) -> None:
        called: list[int] = []

        def run_fn(argv: list[str]) -> subprocess.CompletedProcess:
            called.append(1)
            return _ok()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"MULLVAD_ROTATE": ""}, clear=False):
                ok = maybe_rotate_exit(
                    cities=("nyc",),
                    log_path=Path(tmp) / "rotations.jsonl",
                    run_fn=run_fn,
                    bin_path="/usr/bin/mullvad",
                )
            self.assertFalse(ok)
            self.assertEqual(called, [])

    def test_maybe_rotate_runs_when_enabled(self) -> None:
        calls: list[list[str]] = []
        connected = {"city": "nyc"}

        def run_fn(argv: list[str]) -> subprocess.CompletedProcess:
            calls.append(argv[1:])
            if argv[1] == "status":
                return _ok(f"Connected\n    Relay: us-{connected['city']}-wg-001\n")
            if argv[1:4] == ["relay", "set", "location"]:
                connected["city"] = argv[5]
            return _ok()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"MULLVAD_ROTATE": "1"}, clear=False):
                ok = maybe_rotate_exit(
                    cities=("nyc", "lax"),
                    log_path=Path(tmp) / "rotations.jsonl",
                    run_fn=run_fn,
                    bin_path="/usr/bin/mullvad",
                    fetch_ip_fn=lambda: None,
                )
            self.assertTrue(ok)
            self.assertIn(["connect", "--wait"], calls)

    def test_unchanged_public_ip_fails(self) -> None:
        connected = {"city": "nyc"}

        def run_fn(argv: list[str]) -> subprocess.CompletedProcess:
            if argv[1] == "status":
                return _ok(f"Connected\n    Relay: us-{connected['city']}-wg-001\n")
            if argv[1:4] == ["relay", "set", "location"]:
                connected["city"] = argv[5]
            return _ok()

        with tempfile.TemporaryDirectory() as tmp:
            ok = rotate_exit(
                cities=("nyc", "lax"),
                log_path=Path(tmp) / "rotations.jsonl",
                run_fn=run_fn,
                bin_path="/usr/bin/mullvad",
                fetch_ip_fn=lambda: "1.2.3.4",
            )
            self.assertFalse(ok)

    def test_parse_status_typing(self) -> None:
        self.assertIsInstance(parse_status(""), MullvadStatus)

    def test_gluetun_rotate_put_and_confirm(self) -> None:
        calls: list[tuple[str, str, dict | None]] = []
        ips = ["9.9.9.9", "8.8.8.8"]

        def request_fn(method: str, url: str, payload: dict | None = None) -> tuple[int, str]:
            calls.append((method, url, payload))
            if url.endswith("/v1/vpn/status"):
                return 200, '{"status":"running"}'
            if url.endswith("/v1/vpn/settings") and method == "GET":
                after_put = any(c[0] == "PUT" for c in calls)
                cities = ["Los Angeles CA"] if after_put else ["New York NY"]
                return 200, json.dumps({"provider": {"server_selection": {"cities": cities}}})
            if method == "PUT":
                return 200, "{}"
            return 404, ""

        with tempfile.TemporaryDirectory() as tmp:
            ok = rotate_exit_gluetun(
                cities=("nyc", "lax"),
                log_path=Path(tmp) / "rotations.jsonl",
                request_fn=request_fn,
                fetch_ip_fn=lambda: ips.pop(0) if ips else "8.8.8.8",
                base_url="http://127.0.0.1:8000",
            )
            self.assertTrue(ok)
            put = [c for c in calls if c[0] == "PUT"][0]
            self.assertEqual(put[2], {"provider": {"server_selection": {"cities": ["Los Angeles CA"]}}})

    def test_rotate_exit_dispatches_gluetun(self) -> None:
        with patch("tools.catalog.mullvad_rotate.rotate_exit_gluetun", return_value=True) as mocked:
            with patch.dict("os.environ", {"MULLVAD_BACKEND": "gluetun"}, clear=False):
                ok = rotate_exit(cities=("nyc",), dry_run=True)
            self.assertTrue(ok)
            mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
