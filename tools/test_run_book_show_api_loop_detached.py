#!/usr/bin/env python3
"""Unit tests for tools/run_book_show_api_loop_detached.py
(run: python tools/test_run_book_show_api_loop_detached.py)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.run_book_show_api_loop_detached import (  # noqa: E402
    LOOP_NAME,
    LOOP_SCRIPT,
    is_loop_pid,
    native_loop_env,
    pid_is_alive,
    read_pid,
    start_native_loop,
    write_pid,
)


class TestPidfile(unittest.TestCase):
    def test_write_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loop.pid"
            write_pid(12345, path)
            self.assertEqual(read_pid(path), 12345)

    def test_read_pid_missing_and_garbage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loop.pid"
            self.assertIsNone(read_pid(path))
            path.write_text("nope\n", encoding="utf-8")
            self.assertIsNone(read_pid(path))

    def test_pid_is_alive_self(self) -> None:
        self.assertTrue(pid_is_alive(os.getpid()))
        self.assertFalse(pid_is_alive(0))

    def test_is_loop_pid_rejects_this_process(self) -> None:
        self.assertNotIn(LOOP_NAME, "python unittest")
        self.assertFalse(is_loop_pid(os.getpid()))

    def test_is_loop_pid_accepts_matching_command(self) -> None:
        with patch(
            "tools.run_book_show_api_loop_detached.pid_command",
            return_value="caffeinate -i /tmp/tools/run_book_show_api_loop.sh",
        ):
            with patch("tools.run_book_show_api_loop_detached.pid_is_alive", return_value=True):
                self.assertTrue(is_loop_pid(9))


class TestStartNative(unittest.TestCase):
    def test_native_env_drops_gluetun(self) -> None:
        env = native_loop_env(
            {
                "PATH": "/usr/bin",
                "MULLVAD_BACKEND": "gluetun",
                "GLUETUN_URL": "http://127.0.0.1:8000",
                "VPN_STATUS_URL": "http://127.0.0.1:8000/v1/vpn/status",
            }
        )
        self.assertEqual(env.get("MULLVAD_ROTATE"), "1")
        self.assertNotIn("MULLVAD_BACKEND", env)
        self.assertNotIn("GLUETUN_URL", env)
        self.assertNotIn("VPN_STATUS_URL", env)

    def test_start_native_loop_writes_pid_and_avoids_gluetun(self) -> None:
        seen: dict = {}

        class FakeProc:
            pid = 4321

        def fake_popen(argv, **kwargs):
            seen["argv"] = argv
            seen["env"] = kwargs.get("env")
            seen["cwd"] = kwargs.get("cwd")
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_book_show_api_loop.log"
            pidfile = Path(tmp) / "run_book_show_api_loop.pid"
            with patch("tools.run_book_show_api_loop_detached.pid_path", return_value=pidfile):
                pid = start_native_loop(
                    popen_fn=fake_popen,
                    log=log,
                    env={"PATH": "/usr/bin", "MULLVAD_BACKEND": "gluetun"},
                )
            self.assertEqual(pid, 4321)
            self.assertEqual(read_pid(pidfile), 4321)
            self.assertEqual(seen["argv"][:2], ["caffeinate", "-i"])
            self.assertEqual(seen["argv"][2], str(LOOP_SCRIPT))
            self.assertNotIn("GLUETUN_URL", seen["env"])
            self.assertNotEqual(seen["env"].get("MULLVAD_BACKEND"), "gluetun")
            self.assertEqual(seen["env"].get("MULLVAD_ROTATE"), "1")


if __name__ == "__main__":
    unittest.main()
