#!/usr/bin/env python3
"""Start or heal Xvfb :99 in the scrape container.

Headed Chrome needs a working X display. Docker has no window server, so
entrypoint and the ISBN loop call `ensure` before discover. A leftover
`/tmp/.X99-lock` after Xvfb dies or the container restarts is not "already
running" — it is stale. This module clears that lock, restarts Xvfb, and
refuses to continue if :99 still does not answer.

Native Mac: no-op (Aqua is the display) unless `ENSURE_XVFB=1`.

Usage:
    python tools/catalog/ensure_xvfb.py
    python tools/catalog/ensure_xvfb.py ensure
    python tools/catalog/ensure_xvfb.py status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_DISPLAY = 99
ENV_FORCE = "ENSURE_XVFB"
DOCKERENV = Path("/.dockerenv")
WAIT_SECONDS = 5.0
WAIT_POLL = 0.1
XVFB_ARGS = ("-screen", "0", "1280x800x24", "-ac", "+extension", "RANDR")
DOWN_MESSAGE = (
    "[display] :{display} down — headed Chrome will exit. Not a Goodreads or VPN problem."
)

PidAliveFn = Callable[[int], bool]
IsXvfbFn = Callable[[int], bool]
StartFn = Callable[["DisplayPaths"], int | None]


@dataclass(frozen=True)
class DisplayPaths:
    display: int = DEFAULT_DISPLAY
    lock: Path = Path(f"/tmp/.X{DEFAULT_DISPLAY}-lock")
    socket: Path = Path(f"/tmp/.X11-unix/X{DEFAULT_DISPLAY}")
    log: Path = Path("/tmp/xvfb.log")
    dockerenv: Path = DOCKERENV


def default_paths(display: int = DEFAULT_DISPLAY, tmp: Path | None = None) -> DisplayPaths:
    if tmp is None:
        return DisplayPaths(
            display=display,
            lock=Path(f"/tmp/.X{display}-lock"),
            socket=Path(f"/tmp/.X11-unix/X{display}"),
            log=Path("/tmp/xvfb.log"),
        )
    return DisplayPaths(
        display=display,
        lock=tmp / f".X{display}-lock",
        socket=tmp / ".X11-unix" / f"X{display}",
        log=tmp / "xvfb.log",
        dockerenv=tmp / "dockerenv",
    )


def should_manage(*, dockerenv: Path | None = None, env: dict[str, str] | None = None) -> bool:
    value = (env or os.environ).get(ENV_FORCE, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    path = dockerenv if dockerenv is not None else DOCKERENV
    return path.exists()


def read_lock_pid(lock: Path) -> int | None:
    if not lock.exists():
        return None
    try:
        text = lock.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None
    token = text.split()[0]
    try:
        return int(token)
    except ValueError:
        return None


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def pid_is_xvfb(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except OSError:
        return True
    return "Xvfb" in raw or "xvfb" in raw.lower()


def _is_socket(path: Path) -> bool:
    try:
        return path.is_socket()
    except OSError:
        return False


def display_status(
    paths: DisplayPaths,
    *,
    pid_alive_fn: PidAliveFn | None = None,
    is_xvfb_fn: IsXvfbFn | None = None,
) -> str:
    """`alive` | `stale` | `dead`."""
    alive_fn = pid_alive_fn or pid_is_alive
    xvfb_fn = is_xvfb_fn or pid_is_xvfb
    lock_pid = read_lock_pid(paths.lock)
    socket_ok = _is_socket(paths.socket)
    if (
        lock_pid is not None
        and alive_fn(lock_pid)
        and xvfb_fn(lock_pid)
        and socket_ok
    ):
        return "alive"
    if paths.lock.exists() or paths.socket.exists():
        return "stale"
    return "dead"


def clear_stale(
    paths: DisplayPaths,
    *,
    pid_alive_fn: PidAliveFn | None = None,
    is_xvfb_fn: IsXvfbFn | None = None,
) -> None:
    alive_fn = pid_alive_fn or pid_is_alive
    xvfb_fn = is_xvfb_fn or pid_is_xvfb
    pid = read_lock_pid(paths.lock)
    if pid is None:
        reason = "no pid"
    elif not alive_fn(pid):
        reason = f"pid={pid} (dead)"
    elif not xvfb_fn(pid):
        reason = f"pid={pid} (not Xvfb)"
    else:
        reason = f"pid={pid} (no socket)"
    print(f"[display] stale :{paths.display} lock {reason} — cleared")
    paths.lock.unlink(missing_ok=True)
    try:
        paths.socket.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[display] could not remove socket {paths.socket}: {exc}")


def start_xvfb(paths: DisplayPaths, *, start_fn: StartFn | None = None) -> int | None:
    if start_fn is not None:
        return start_fn(paths)
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    paths.socket.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["Xvfb", f":{paths.display}", *XVFB_ARGS]
    with paths.log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def wait_alive(
    paths: DisplayPaths,
    *,
    timeout: float = WAIT_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    pid_alive_fn: PidAliveFn | None = None,
    is_xvfb_fn: IsXvfbFn | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if display_status(paths, pid_alive_fn=pid_alive_fn, is_xvfb_fn=is_xvfb_fn) == "alive":
            return True
        sleep_fn(WAIT_POLL)
    return display_status(paths, pid_alive_fn=pid_alive_fn, is_xvfb_fn=is_xvfb_fn) == "alive"


def _print_log_tail(log: Path, lines: int = 8) -> None:
    if not log.exists():
        return
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[display] could not read {log}: {exc}")
        return
    tail = text.strip().splitlines()[-lines:]
    if not tail:
        return
    print("[display] xvfb.log:")
    for line in tail:
        print(f"  {line}")


def ensure(
    paths: DisplayPaths | None = None,
    *,
    should_manage_fn: Callable[[], bool] | None = None,
    start_fn: StartFn | None = None,
    pid_alive_fn: PidAliveFn | None = None,
    is_xvfb_fn: IsXvfbFn | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    timeout: float = WAIT_SECONDS,
) -> int:
    paths = paths or default_paths()
    os.environ.setdefault("DISPLAY", f":{paths.display}")
    manage = should_manage_fn() if should_manage_fn is not None else should_manage(dockerenv=paths.dockerenv)
    if not manage:
        return 0

    status = display_status(paths, pid_alive_fn=pid_alive_fn, is_xvfb_fn=is_xvfb_fn)
    if status == "alive":
        print(f"[display] :{paths.display} alive pid={read_lock_pid(paths.lock)}")
        return 0
    if status == "stale":
        clear_stale(paths, pid_alive_fn=pid_alive_fn, is_xvfb_fn=is_xvfb_fn)

    started = start_xvfb(paths, start_fn=start_fn)
    if wait_alive(
        paths,
        timeout=timeout,
        sleep_fn=sleep_fn,
        pid_alive_fn=pid_alive_fn,
        is_xvfb_fn=is_xvfb_fn,
    ):
        print(f"[display] started Xvfb :{paths.display} pid={read_lock_pid(paths.lock) or started}")
        return 0

    _print_log_tail(paths.log)
    print(DOWN_MESSAGE.format(display=paths.display))
    return 1


def status_line(
    paths: DisplayPaths,
    *,
    should_manage_fn: Callable[[], bool] | None = None,
    pid_alive_fn: PidAliveFn | None = None,
    is_xvfb_fn: IsXvfbFn | None = None,
) -> tuple[int, str]:
    manage = should_manage_fn() if should_manage_fn is not None else should_manage(dockerenv=paths.dockerenv)
    if not manage:
        return 0, "[display] skip (not managing Xvfb)"
    state = display_status(paths, pid_alive_fn=pid_alive_fn, is_xvfb_fn=is_xvfb_fn)
    pid = read_lock_pid(paths.lock)
    suffix = f" pid={pid}" if pid is not None else ""
    line = f"[display] :{paths.display} {state}{suffix}"
    return (0 if state == "alive" else 1), line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("ensure", "status"),
        default="ensure",
    )
    args = parser.parse_args(argv)
    paths = default_paths()
    if args.command == "status":
        rc, line = status_line(paths)
        print(line)
        return rc
    return ensure(paths)


if __name__ == "__main__":
    raise SystemExit(main())
