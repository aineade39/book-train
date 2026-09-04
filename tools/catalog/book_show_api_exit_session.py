#!/usr/bin/env python3
"""Prepare a Goodreads Chrome session for one VPN exit: fresh profile, human warmup, discover.

After a Mullvad rotate the ISBN loop wipes `book_show_api_chunked`, visits
goodreads.com, opens a list that contains the discover book (LOTR / id 33),
clicks through (or falls back to the book URL), and only then starts the API
scrape in that same profile.

Discover failures rotate to the next city immediately (no 15-minute sleep).
A full walk of the city pool pauses once; a second full walk hard-stops so
the loop cannot spin.

Usage:
    python tools/catalog/book_show_api_exit_session.py prepare-exit --harness-root ~/dev/scrape-harness
    python tools/catalog/book_show_api_exit_session.py next-fail-action --consecutive 8 --cycles 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.mullvad_rotate import DEFAULT_CITIES, parse_cities  # noqa: E402
from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_HARNESS_ROOT = _REPO.parent / "scrape-harness"
HOME_URL = "https://www.goodreads.com/"
DISCOVER_URL = "https://www.goodreads.com/book/show/33.The_Lord_of_the_Rings"
# Best Fantasy Books — LOTR (book 33) is a fixture on page 1.
DISCOVER_LIST_URL = "https://www.goodreads.com/list/show/367.Best_Fantasy_Books"
DISCOVER_BOOK_HREF = "/book/show/33"
CHUNKED_PROFILE_ID = "book_show_api_chunked"
NAV_DELAY_MS_MIN = 400
NAV_DELAY_MS_MAX = 1_200
GOTO_TIMEOUT_MS = 15_000
DISCOVER_SUBPROCESS_TIMEOUT = 90
DEFAULT_FAIL_STATE_NAME = "discover_fail_state.json"
DEFAULT_MAX_CYCLES = 2
DEFAULT_CYCLE_SLEEP_SECONDS = 900
ACTION_ROTATE = "rotate"
ACTION_CYCLE_SLEEP = "cycle_sleep"
ACTION_HARD_STOP = "hard_stop"
PREPARE_OK = 0
PREPARE_FAIL = 1
PREPARE_HARD_STOP = 2
PREPARE_DISPLAY_FAIL = 4
DISPLAY_FAILURE_MARKERS = (
    "Missing X server",
    "without having a XServer",
    "ozone_platform_x11",
)


@dataclass(frozen=True)
class DiscoverFailState:
    consecutive: int = 0
    cycles: int = 0


def rotate_limit(cities: tuple[str, ...] | None = None) -> int:
    pool = cities if cities is not None else parse_cities()
    return max(1, len(pool) if pool else len(DEFAULT_CITIES))


def next_fail_action(
    consecutive: int,
    cycles: int,
    *,
    limit: int | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> str:
    """What to do after a discover failure (consecutive already includes this fail)."""
    cap = limit if limit is not None else rotate_limit()
    if consecutive < cap:
        return ACTION_ROTATE
    if cycles + 1 >= max_cycles:
        return ACTION_HARD_STOP
    return ACTION_CYCLE_SLEEP


def apply_fail(state: DiscoverFailState, *, limit: int | None = None, max_cycles: int = DEFAULT_MAX_CYCLES) -> tuple[DiscoverFailState, str]:
    consecutive = state.consecutive + 1
    action = next_fail_action(consecutive, state.cycles, limit=limit, max_cycles=max_cycles)
    if action == ACTION_ROTATE:
        return DiscoverFailState(consecutive=consecutive, cycles=state.cycles), action
    if action == ACTION_CYCLE_SLEEP:
        return DiscoverFailState(consecutive=0, cycles=state.cycles + 1), action
    return DiscoverFailState(consecutive=consecutive, cycles=state.cycles + 1), action


def reset_fail_state() -> DiscoverFailState:
    return DiscoverFailState()


def default_fail_state_path(catalog_dir: Path | None = None) -> Path:
    if catalog_dir is not None:
        return catalog_dir / DEFAULT_FAIL_STATE_NAME
    return Path(catalog_goodreads(DEFAULT_FAIL_STATE_NAME))


def load_fail_state(path: Path) -> DiscoverFailState:
    if not path.exists():
        return DiscoverFailState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DiscoverFailState()
    try:
        return DiscoverFailState(consecutive=int(data.get("consecutive", 0)), cycles=int(data.get("cycles", 0)))
    except (TypeError, ValueError):
        return DiscoverFailState()


def save_fail_state(path: Path, state: DiscoverFailState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "consecutive": state.consecutive,
        "cycles": state.cycles,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def clear_fail_state(path: Path) -> None:
    path.unlink(missing_ok=True)


def discover_command(
    harness_root: Path,
    *,
    fresh_browser: bool = True,
    skip_vpn_check: bool = False,
) -> list[str]:
    harness_bin = harness_root / ".venv" / "bin" / "scrape-harness"
    cmd = [
        str(harness_bin),
        "api",
        "discover",
        "goodreads",
        "--url",
        DISCOVER_URL,
        "--user-data-id",
        CHUNKED_PROFILE_ID,
        "--warmup-url",
        HOME_URL,
        "--warmup-url",
        DISCOVER_LIST_URL,
        "--click-href-contains",
        DISCOVER_BOOK_HREF,
        "--nav-delay-ms-min",
        str(NAV_DELAY_MS_MIN),
        "--nav-delay-ms-max",
        str(NAV_DELAY_MS_MAX),
        "--goto-timeout-ms",
        str(GOTO_TIMEOUT_MS),
    ]
    if fresh_browser:
        cmd.append("--fresh-browser")
    else:
        cmd.append("--keep-profile")
    if skip_vpn_check:
        cmd.append("--skip-vpn-check")
    return cmd


def wipe_chunked_profile(harness_root: Path) -> None:
    chunked = harness_root / ".data" / "browser_profiles" / "goodreads" / "book_show_api_chunked"
    if chunked.exists():
        shutil.rmtree(chunked, ignore_errors=True)
        print(f"[exit_session] wiped browser profile {chunked}")
    lock_path = harness_root / ".data" / "locks" / "goodreads.lock"
    if lock_path.exists():
        try:
            lock_path.unlink()
            print(f"[exit_session] cleared stale site lock {lock_path}")
        except OSError as exc:
            print(f"[exit_session] could not clear site lock {lock_path}: {exc}")


def is_display_failure(text: str) -> bool:
    """True when Chrome/Playwright died because :99 (or any X display) is gone."""
    return any(marker in text for marker in DISPLAY_FAILURE_MARKERS)


def _proc_output(proc: object) -> str:
    parts: list[str] = []
    for attr in ("stdout", "stderr"):
        value = getattr(proc, attr, None)
        if isinstance(value, bytes):
            parts.append(value.decode("utf-8", errors="replace"))
        elif isinstance(value, str) and value:
            parts.append(value)
    return "".join(parts)


def prepare_exit(
    harness_root: Path,
    *,
    fresh_browser: bool = True,
    timeout: int = DISCOVER_SUBPROCESS_TIMEOUT,
    run_fn=None,
) -> bool:
    """Wipe the chunked profile (optional) and run warmup+discover in that dir.

    Returns True on success. On failure, `cmd_prepare_exit` maps a Chrome
    missing-X error to `PREPARE_DISPLAY_FAIL` (4) so the loop can heal
    Xvfb instead of rotating Mullvad.
    """
    result = prepare_exit_result(
        harness_root,
        fresh_browser=fresh_browser,
        timeout=timeout,
        run_fn=run_fn,
    )
    return result.ok


@dataclass(frozen=True)
class PrepareExitResult:
    ok: bool
    display_failure: bool = False
    output: str = ""


def prepare_exit_result(
    harness_root: Path,
    *,
    fresh_browser: bool = True,
    timeout: int = DISCOVER_SUBPROCESS_TIMEOUT,
    run_fn=None,
) -> PrepareExitResult:
    if fresh_browser:
        wipe_chunked_profile(harness_root)
    harness_bin = harness_root / ".venv" / "bin" / "scrape-harness"
    if not harness_bin.exists():
        print(f"[exit_session] scrape-harness binary missing at {harness_bin}")
        return PrepareExitResult(ok=False)
    cmd = discover_command(harness_root, fresh_browser=fresh_browser)
    print(f"[exit_session] prepare-exit: {' '.join(cmd)}")
    runner = run_fn or (
        lambda argv: subprocess.run(
            argv,
            cwd=harness_root,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    )
    try:
        proc = runner(cmd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        text = str(exc)
        print(f"[exit_session] discover failed: {exc}")
        return PrepareExitResult(ok=False, display_failure=is_display_failure(text), output=text)
    output = _proc_output(proc)
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    rc = proc.returncode if proc is not None else 1
    if rc != 0:
        print(f"[exit_session] discover exited {rc}")
        display_fail = is_display_failure(output)
        if display_fail:
            print("[exit_session] Chrome exited: no X display (not a discover/VPN miss)")
        return PrepareExitResult(ok=False, display_failure=display_fail, output=output)
    return PrepareExitResult(ok=True, output=output)


def cmd_prepare_exit(args: argparse.Namespace) -> int:
    result = prepare_exit_result(Path(args.harness_root), fresh_browser=not args.keep_profile)
    if result.ok:
        return PREPARE_OK
    if result.display_failure:
        return PREPARE_DISPLAY_FAIL
    return PREPARE_FAIL


def cmd_next_fail_action(args: argparse.Namespace) -> int:
    path = Path(args.state_path) if args.state_path else default_fail_state_path(Path(args.catalog_dir) if args.catalog_dir else None)
    state = load_fail_state(path)
    if args.consecutive is not None:
        state = DiscoverFailState(consecutive=args.consecutive, cycles=args.cycles if args.cycles is not None else state.cycles)
    if args.reset:
        clear_fail_state(path)
        print(f"[exit_session] cleared discover-fail state ({path})")
        return PREPARE_OK
    new_state, action = apply_fail(
        state,
        limit=args.limit,
        max_cycles=args.max_cycles,
    )
    save_fail_state(path, new_state)
    print(
        f"[exit_session] discover fail action={action} "
        f"consecutive={new_state.consecutive} cycles={new_state.cycles}"
    )
    print(action)
    if action == ACTION_HARD_STOP:
        return PREPARE_HARD_STOP
    if action == ACTION_CYCLE_SLEEP:
        print(f"[exit_session] full city pool failed — sleep {args.cycle_sleep}s before rotating")
        return 3
    return PREPARE_FAIL


def cmd_reset_fail(args: argparse.Namespace) -> int:
    path = Path(args.state_path) if args.state_path else default_fail_state_path(Path(args.catalog_dir) if args.catalog_dir else None)
    clear_fail_state(path)
    print(f"[exit_session] cleared discover-fail state ({path})")
    return PREPARE_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-exit", help="Fresh (or kept) chunked profile + home/list/discover")
    prepare.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS_ROOT)
    prepare.add_argument(
        "--keep-profile",
        action="store_true",
        help="Do not wipe the chunked Chrome profile (same exit, retry discover).",
    )
    prepare.set_defaults(func=cmd_prepare_exit)

    nxt = sub.add_parser("next-fail-action", help="Record a discover failure and print rotate/cycle_sleep/hard_stop")
    nxt.add_argument("--catalog-dir", type=Path, default=None)
    nxt.add_argument("--state-path", type=Path, default=None)
    nxt.add_argument("--consecutive", type=int, default=None, help="Override loaded consecutive count before this fail")
    nxt.add_argument("--cycles", type=int, default=None)
    nxt.add_argument("--limit", type=int, default=None)
    nxt.add_argument("--max-cycles", type=int, default=int(os.environ.get("DISCOVER_FAIL_MAX_CYCLES", DEFAULT_MAX_CYCLES)))
    nxt.add_argument("--cycle-sleep", type=int, default=int(os.environ.get("DISCOVER_FAIL_CYCLE_SLEEP", DEFAULT_CYCLE_SLEEP_SECONDS)))
    nxt.add_argument("--reset", action="store_true", help="Clear state instead of recording a fail")
    nxt.set_defaults(func=cmd_next_fail_action)

    reset = sub.add_parser("reset-fail", help="Clear discover-fail state after a successful prepare-exit")
    reset.add_argument("--catalog-dir", type=Path, default=None)
    reset.add_argument("--state-path", type=Path, default=None)
    reset.set_defaults(func=cmd_reset_fail)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
