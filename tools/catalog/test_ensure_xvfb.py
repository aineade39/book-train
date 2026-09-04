#!/usr/bin/env python3
"""Unit tests for tools/catalog/ensure_xvfb.py
(run: python tools/catalog/test_ensure_xvfb.py).

Does not start a real Xvfb — lock/socket/pid checks are injected."""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ensure_xvfb import (  # noqa: E402
    DOWN_MESSAGE,
    DisplayPaths,
    clear_stale,
    default_paths,
    display_status,
    ensure,
    main,
    read_lock_pid,
    should_manage,
    status_line,
)


def _make_socket(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
    finally:
        sock.close()


def _paths(tmp: Path) -> DisplayPaths:
    return default_paths(tmp=tmp)


class TestShouldManage(unittest.TestCase):
    def test_false_without_dockerenv_or_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(should_manage(dockerenv=Path(tmp) / "missing", env={}))

    def test_true_when_dockerenv_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dockerenv = Path(tmp) / "dockerenv"
            dockerenv.write_text("", encoding="utf-8")
            self.assertTrue(should_manage(dockerenv=dockerenv, env={}))

    def test_true_when_env_forced(self) -> None:
        self.assertTrue(should_manage(dockerenv=Path("/no/such/dockerenv"), env={"ENSURE_XVFB": "1"}))


class TestDisplayStatus(unittest.TestCase):
    def test_dead_when_nothing_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            self.assertEqual(display_status(paths, pid_alive_fn=lambda _p: False, is_xvfb_fn=lambda _p: True), "dead")

    def test_alive_requires_live_xvfb_pid_and_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("4242\n", encoding="utf-8")
            _make_socket(paths.socket)
            self.assertEqual(display_status(paths, pid_alive_fn=lambda p: p == 4242, is_xvfb_fn=lambda _p: True), "alive")

    def test_stale_when_lock_pid_dead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("99\n", encoding="utf-8")
            _make_socket(paths.socket)
            self.assertEqual(display_status(paths, pid_alive_fn=lambda _p: False, is_xvfb_fn=lambda _p: True), "stale")

    def test_stale_when_socket_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("4242\n", encoding="utf-8")
            self.assertEqual(display_status(paths, pid_alive_fn=lambda _p: True, is_xvfb_fn=lambda _p: True), "stale")

    def test_stale_when_pid_not_xvfb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("4242\n", encoding="utf-8")
            _make_socket(paths.socket)
            self.assertEqual(display_status(paths, pid_alive_fn=lambda _p: True, is_xvfb_fn=lambda _p: False), "stale")

    def test_read_lock_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lock"
            lock.write_text("    17\n", encoding="utf-8")
            self.assertEqual(read_lock_pid(lock), 17)


class TestClearStale(unittest.TestCase):
    def test_removes_lock_and_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("99\n", encoding="utf-8")
            _make_socket(paths.socket)
            clear_stale(paths, pid_alive_fn=lambda _p: False, is_xvfb_fn=lambda _p: True)
            self.assertFalse(paths.lock.exists())
            self.assertFalse(paths.socket.exists())


class TestEnsure(unittest.TestCase):
    def test_skip_outside_container(self) -> None:
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            started = []
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = ensure(
                    paths,
                    should_manage_fn=lambda: False,
                    start_fn=lambda p: started.append(p) or 1,
                )
            self.assertEqual(rc, 0)
            self.assertEqual(started, [])
            self.assertEqual(buf.getvalue(), "")

    def test_live_pid_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("4242\n", encoding="utf-8")
            _make_socket(paths.socket)
            started = []
            rc = ensure(
                paths,
                should_manage_fn=lambda: True,
                start_fn=lambda p: started.append(p) or 1,
                pid_alive_fn=lambda p: p == 4242,
                is_xvfb_fn=lambda _p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(started, [])

    def test_stale_lock_cleared_then_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("99\n", encoding="utf-8")
            _make_socket(paths.socket)

            def start(p: DisplayPaths) -> int:
                p.lock.write_text("4242\n", encoding="utf-8")
                _make_socket(p.socket)
                return 4242

            rc = ensure(
                paths,
                should_manage_fn=lambda: True,
                start_fn=start,
                pid_alive_fn=lambda p: p == 4242,
                is_xvfb_fn=lambda _p: True,
                sleep_fn=lambda _s: None,
                timeout=0.3,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(read_lock_pid(paths.lock), 4242)

    def test_start_failure_prints_display_message(self) -> None:
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.log.write_text("Fatal server error:\nServer is already active\n", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = ensure(
                    paths,
                    should_manage_fn=lambda: True,
                    start_fn=lambda _p: None,
                    pid_alive_fn=lambda _p: False,
                    is_xvfb_fn=lambda _p: True,
                    sleep_fn=lambda _s: None,
                    timeout=0.0,
                )
            self.assertEqual(rc, 1)
            out = buf.getvalue()
            self.assertIn(DOWN_MESSAGE.format(display=99), out)
            self.assertIn("Server is already active", out)


class TestStatusAndCli(unittest.TestCase):
    def test_status_alive_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("7\n", encoding="utf-8")
            _make_socket(paths.socket)
            rc, line = status_line(
                paths,
                should_manage_fn=lambda: True,
                pid_alive_fn=lambda _p: True,
                is_xvfb_fn=lambda _p: True,
            )
            self.assertEqual(rc, 0)
            self.assertIn("alive", line)
            self.assertIn("pid=7", line)

    def test_status_stale_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            paths.lock.write_text("7\n", encoding="utf-8")
            rc, line = status_line(
                paths,
                should_manage_fn=lambda: True,
                pid_alive_fn=lambda _p: False,
                is_xvfb_fn=lambda _p: True,
            )
            self.assertEqual(rc, 1)
            self.assertIn("stale", line)

    def test_cli_ensure_skips_on_host(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ENSURE_XVFB"}
        with patch.dict(os.environ, env, clear=True):
            if Path("/.dockerenv").exists():
                self.skipTest("running inside docker")
            self.assertEqual(main(["ensure"]), 0)


if __name__ == "__main__":
    unittest.main()
