#!/usr/bin/env python3
"""Start/stop the book_show_api loop in Docker (or detect a leftover native loop).

Cursor Agent-spawned shells get aborted after ~3h. `start` runs
`docker compose up -d` and a host `caffeinate` (ppid 1), then returns.
`start-native` stops compose and launches `caffeinate -i` + the loop in a
new session (not an Agent shell).

Usage:
    python tools/run_book_show_api_loop_detached.py prepare
    python tools/run_book_show_api_loop_detached.py start
    python tools/run_book_show_api_loop_detached.py start-native
    python tools/run_book_show_api_loop_detached.py status
    python tools/run_book_show_api_loop_detached.py stop
    python tools/run_book_show_api_loop_detached.py log
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.container_scrape import (  # noqa: E402
    COMPOSE_DIR,
    DISK_CLEANUP_ROOT,
    ITEM_ID,
    backup_verified,
    caffeinate_pid_path,
    compose_bind_matches_host,
    compose_bind_source,
    ensure_harness_bind_files,
    host_data_root,
    jsonl_stat,
    read_marker,
    write_marker,
)
from tools.paths import catalog_goodreads  # noqa: E402

LOOP_SCRIPT = _REPO / "tools" / "run_book_show_api_loop.sh"
LOOP_NAME = "run_book_show_api_loop.sh"
DEFAULT_PID_NAME = "run_book_show_api_loop.pid"
DEFAULT_LOG_NAME = "run_book_show_api_loop.log"
COMPOSE_PROJECT = "book-show-api"


def pid_path() -> Path:
    return Path(catalog_goodreads(DEFAULT_PID_NAME))


def log_path() -> Path:
    return Path(catalog_goodreads(DEFAULT_LOG_NAME))


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def pid_command(pid: int) -> str:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return (proc.stdout or "").strip()


def is_loop_pid(pid: int) -> bool:
    if not pid_is_alive(pid):
        return False
    return LOOP_NAME in pid_command(pid)


def read_pid(path: Path | None = None) -> int | None:
    target = path or pid_path()
    if not target.exists():
        return None
    raw = target.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def write_pid(pid: int, path: Path | None = None) -> None:
    target = path or pid_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(pid), encoding="utf-8")


def running_native_pid() -> int | None:
    pid = read_pid()
    if pid is not None and is_loop_pid(pid):
        return pid
    return None


def _compose_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("BOOK_SPINES_DATA", str(host_data_root()))
    env.setdefault("HARNESS_HOST", str((_REPO.parent / "scrape-harness").resolve()))
    env.setdefault("HOME", str(Path.home()))
    try:
        env["BOOK_TRAIN_REV"] = subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        env.setdefault("BOOK_TRAIN_REV", "unknown")
    harness = Path(env["HARNESS_HOST"])
    try:
        env["HARNESS_REV"] = subprocess.check_output(
            ["git", "-C", str(harness), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        env.setdefault("HARNESS_REV", "unknown")
    return env


def _compose(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-p", COMPOSE_PROJECT, "-f", str(COMPOSE_DIR / "docker-compose.yml"), *args]
    try:
        return subprocess.run(cmd, cwd=str(COMPOSE_DIR), env=_compose_env(), check=check, text=True)
    except FileNotFoundError:
        print("[detached] docker not on PATH — start Docker Desktop first", file=sys.stderr)
        if check:
            raise SystemExit(1)
        return subprocess.CompletedProcess(cmd, 1, "", "docker not found")


def _compose_running() -> bool:
    try:
        proc = subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                COMPOSE_PROJECT,
                "-f",
                str(COMPOSE_DIR / "docker-compose.yml"),
                "ps",
                "-q",
                "scrape",
            ],
            cwd=str(COMPOSE_DIR),
            env=_compose_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return bool((proc.stdout or "").strip())


def _stop_native() -> None:
    pid = running_native_pid()
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    for _ in range(20):
        if not pid_is_alive(pid):
            break
        time.sleep(0.15)
    pid_path().unlink(missing_ok=True)


def _start_host_caffeinate() -> int:
    existing = read_pid(caffeinate_pid_path())
    if existing is not None and pid_is_alive(existing) and "caffeinate" in pid_command(existing):
        return existing
    proc = subprocess.Popen(
        ["caffeinate", "-i"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    write_pid(proc.pid, caffeinate_pid_path())
    return proc.pid


def _stop_host_caffeinate() -> None:
    path = caffeinate_pid_path()
    pid = read_pid(path)
    if pid is not None and pid_is_alive(pid) and "caffeinate" in pid_command(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    path.unlink(missing_ok=True)


def cmd_prepare() -> int:
    _stop_native()
    if running_native_pid() is not None:
        print("[detached] native loop still running — stop it first", file=sys.stderr)
        return 1
    if _compose_running():
        print("[detached] compose scrape already up — stop it first", file=sys.stderr)
        return 1
    try:
        stat = jsonl_stat()
    except FileNotFoundError as exc:
        print(f"[detached] {exc}", file=sys.stderr)
        return 1
    if not compose_bind_matches_host():
        print(
            f"[detached] compose bind {compose_bind_source()} != host {host_data_root()}",
            file=sys.stderr,
        )
        return 1
    ok, detail = backup_verified()
    if not ok:
        print(f"[detached] backup not verified: {detail}", file=sys.stderr)
        print(
            f"[detached] in {DISK_CLEANUP_ROOT}: "
            f"disk-cleanup nas-copy --item {ITEM_ID} --execute && "
            f"disk-cleanup gdrive-sync --item {ITEM_ID} --execute",
            file=sys.stderr,
        )
        return 1
    harness = Path(os.environ.get("HARNESS_ROOT") or (_REPO.parent / "scrape-harness"))
    ensure_harness_bind_files(harness)
    path = write_marker(run_id=detail, stat=stat)
    print(f"[detached] prepared marker={path} run_id={detail} lines={stat['lines']} bytes={stat['bytes']}")
    return 0


def native_loop_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Host scrape env: Mullvad CLI, not Gluetun."""
    out = dict(env if env is not None else os.environ)
    out.setdefault("MULLVAD_ROTATE", "1")
    if out.get("MULLVAD_BACKEND", "").strip().lower() == "gluetun":
        del out["MULLVAD_BACKEND"]
    out.pop("GLUETUN_URL", None)
    out.pop("VPN_STATUS_URL", None)
    return out


def start_native_loop(
    *,
    popen_fn=None,
    log: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Launch `caffeinate -i` + the loop in a new session. Returns the pid."""
    path = log if log is not None else log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    launch = popen_fn or subprocess.Popen
    handle = path.open("a", encoding="utf-8")
    try:
        proc = launch(
            ["caffeinate", "-i", str(LOOP_SCRIPT)],
            cwd=str(_REPO),
            env=native_loop_env(env),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    finally:
        handle.close()
    write_pid(proc.pid)
    return proc.pid


def cmd_start_native() -> int:
    if running_native_pid() is not None:
        print(f"[detached] native loop already running pid={running_native_pid()}", file=sys.stderr)
        return 1
    cmd_stop()
    pid = start_native_loop()
    print(f"[detached] started native pid={pid} log={log_path()}")
    return 0


def cmd_start() -> int:
    if running_native_pid() is not None:
        print("[detached] native loop is running — stop it before container start", file=sys.stderr)
        return 1
    marker = read_marker()
    if marker is None:
        print("[detached] missing container_migration.json — run prepare first", file=sys.stderr)
        return 1
    if not compose_bind_matches_host():
        print(
            f"[detached] compose bind {compose_bind_source()} != host {host_data_root()}",
            file=sys.stderr,
        )
        return 1
    if _compose_running():
        print(f"[detached] already running compose project={COMPOSE_PROJECT}")
        return 0
    env_file = COMPOSE_DIR / ".env"
    if not env_file.is_file():
        print(f"[detached] missing {env_file} — copy .env.example and add the WireGuard device", file=sys.stderr)
        return 1
    _compose(["up", "-d"])
    cafe = _start_host_caffeinate()
    print(f"[detached] started compose project={COMPOSE_PROJECT} caffeinate={cafe} log={log_path()}")
    return 0


def cmd_status() -> int:
    native = running_native_pid()
    if native is not None:
        print(f"[detached] native loop running pid={native}")
        return 0
    marker = read_marker()
    print(f"[detached] marker={'present' if marker else 'missing'}")
    _compose(["ps"], check=False)
    cafe = read_pid(caffeinate_pid_path())
    if cafe is not None and pid_is_alive(cafe):
        print(f"[detached] caffeinate pid={cafe}")
    else:
        print("[detached] caffeinate not running")
    print(f"[detached] log={log_path()}")
    return 0 if _compose_running() or native is not None else 1


def cmd_stop() -> int:
    _stop_native()
    _stop_host_caffeinate()
    _compose(["stop"], check=False)
    print(f"[detached] stopped compose project={COMPOSE_PROJECT}")
    return 0


def cmd_log(lines: int) -> int:
    path = log_path()
    _compose(["logs", "--no-color", "--tail", str(lines)], check=False)
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if text:
            print("--- host log ---")
            for line in text[-lines:]:
                print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="One-time migration gate: backup verified, write marker.")
    sub.add_parser("start", help="docker compose up -d plus host caffeinate (no-op if already up).")
    sub.add_parser(
        "start-native",
        help="Stop compose; caffeinate -i the host loop in a new session (Mac Mullvad).",
    )
    sub.add_parser("status", help="Print compose / caffeinate / marker state.")
    sub.add_parser("stop", help="Stop compose scrape/vpn, native loop, and host caffeinate.")
    p_log = sub.add_parser("log", help="Print compose logs and the host loop log tail.")
    p_log.add_argument("-n", type=int, default=40)
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        return cmd_prepare()
    if args.cmd == "start":
        return cmd_start()
    if args.cmd == "start-native":
        return cmd_start_native()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "stop":
        return cmd_stop()
    return cmd_log(args.n)


if __name__ == "__main__":
    raise SystemExit(main())
