#!/usr/bin/env python3
"""Rotate the Mullvad exit to the next US city between book_show_api chunks.

Used by `tools/run_book_show_api_loop.sh` during the healthy inter-chunk
cooldown and by `book_show_api_session_health.py` on IP-related spikes.
City-level only (`mullvad relay set location us <code>`); Mullvad picks a
random WireGuard server inside that city.

Opt-in: set `MULLVAD_ROTATE=1`. Override the pool with
`MULLVAD_CITIES="nyc lax chi …"`. Missing binary or a failed connect logs
and returns False — do not abort a healthy scrape.

Usage:
    python tools/catalog/mullvad_rotate.py
    python tools/catalog/mullvad_rotate.py --dry-run
    MULLVAD_ROTATE=1 python tools/catalog/mullvad_rotate.py --require-enabled
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.paths import catalog_goodreads  # noqa: E402

DEFAULT_COUNTRY = "us"
DEFAULT_CITIES = ("nyc", "lax", "chi", "dal", "sea", "atl", "mia", "qas")
ENV_ROTATE = "MULLVAD_ROTATE"
ENV_CITIES = "MULLVAD_CITIES"
ENV_BACKEND = "MULLVAD_BACKEND"
ENV_GLUETUN_URL = "GLUETUN_URL"
DEFAULT_GLUETUN_URL = "http://127.0.0.1:8000"
DEFAULT_LOG_NAME = "mullvad_rotations.jsonl"
GLUETUN_CITY_NAMES = {
    "nyc": "New York NY",
    "lax": "Los Angeles CA",
    "chi": "Chicago IL",
    "dal": "Dallas TX",
    "sea": "Seattle WA",
    "atl": "Atlanta GA",
    "mia": "Miami FL",
    "qas": "Ashburn VA",
}
NAME_TO_CITY = {name.lower(): code for code, name in GLUETUN_CITY_NAMES.items()}
HOSTNAME_CITY_RE = re.compile(r"^us-([a-z]{3})-", re.IGNORECASE)
RELAY_LINE_RE = re.compile(r"Relay:\s+(\S+)", re.IGNORECASE)
MULLVAD_BIN_CANDIDATES = (
    "/usr/local/bin/mullvad",
    "/opt/homebrew/bin/mullvad",
    "/Applications/Mullvad VPN.app/Contents/Resources/mullvad-cli",
)
RunFn = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass(frozen=True)
class MullvadStatus:
    connected: bool
    city: str | None
    hostname: str | None = None
    raw: str = ""


def rotate_backend(env: dict[str, str] | None = None) -> str:
    value = (env or os.environ).get(ENV_BACKEND, "cli").strip().lower()
    return value or "cli"


def gluetun_url(env: dict[str, str] | None = None) -> str:
    return ((env or os.environ).get(ENV_GLUETUN_URL, "") or DEFAULT_GLUETUN_URL).rstrip("/")


def rotate_enabled(env: dict[str, str] | None = None) -> bool:
    value = (env or os.environ).get(ENV_ROTATE, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def parse_cities(raw: str | None = None) -> tuple[str, ...]:
    text = raw if raw is not None else os.environ.get(ENV_CITIES, "")
    parts = [p.strip().lower() for p in re.split(r"[,\s]+", text or "") if p.strip()]
    if not parts:
        return DEFAULT_CITIES
    return tuple(parts)


def next_city(cities: tuple[str, ...], current: str | None) -> str:
    if not cities:
        raise ValueError("city pool is empty")
    if current is None or current not in cities:
        return cities[0]
    return cities[(cities.index(current) + 1) % len(cities)]


def find_mullvad_bin() -> str | None:
    found = shutil.which("mullvad")
    if found:
        return found
    for path in MULLVAD_BIN_CANDIDATES:
        if Path(path).is_file() and os.access(path, os.X_OK):
            return path
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_fn(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=120)


def _city_from_hostname(hostname: str | None) -> str | None:
    if not hostname:
        return None
    match = HOSTNAME_CITY_RE.match(hostname.strip())
    return match.group(1).lower() if match else None


def parse_status(stdout: str) -> MullvadStatus:
    text = stdout.strip()
    if not text:
        return MullvadStatus(connected=False, city=None, raw=text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        state = str(data.get("state") or data.get("Status") or "").lower()
        details = data.get("details") if isinstance(data.get("details"), dict) else data
        location = details.get("location") if isinstance(details, dict) else None
        hostname = None
        city = None
        if isinstance(location, dict):
            hostname = location.get("hostname") or location.get("relay")
            city = location.get("city_code") or location.get("code")
            if isinstance(city, str):
                city = city.lower()
            else:
                city = None
        if not hostname and isinstance(details, dict):
            hostname = details.get("hostname")
        if not city:
            city = _city_from_hostname(str(hostname) if hostname else None)
        connected = state in {"connected", "connecting"} or bool(data.get("connected"))
        return MullvadStatus(
            connected=connected,
            city=city,
            hostname=str(hostname) if hostname else None,
            raw=text,
        )

    lowered = text.lower()
    connected = "disconnected" not in lowered.splitlines()[0] and "connected" in lowered
    hostname = None
    relay_match = RELAY_LINE_RE.search(text)
    if relay_match:
        hostname = relay_match.group(1)
    city = _city_from_hostname(hostname)
    return MullvadStatus(connected=connected, city=city, hostname=hostname, raw=text)


def _run_status(bin_path: str, run_fn: RunFn) -> MullvadStatus:
    json_proc = run_fn([bin_path, "status", "--json"])
    if json_proc.returncode == 0 and json_proc.stdout.strip().startswith("{"):
        return parse_status(json_proc.stdout)
    verbose = run_fn([bin_path, "status", "-v"])
    combined = verbose.stdout or json_proc.stdout or json_proc.stderr
    return parse_status(combined)


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def fetch_public_ip(url: str = "https://api.ipify.org", timeout: float = 8.0) -> str | None:
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed IP-echo URL
            body = response.read().decode("utf-8", errors="replace").strip()
    except (URLError, TimeoutError, OSError):
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body or None
    if isinstance(data, dict) and data.get("ip"):
        return str(data["ip"])
    return body or None


def _gluetun_request(
    method: str,
    url: str,
    *,
    payload: dict | None = None,
    timeout: float = 30.0,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-configured Gluetun URL
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body
    except (URLError, TimeoutError, OSError) as exc:
        return 0, str(exc)


def _city_from_gluetun_settings(body: str) -> str | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    provider = data.get("provider") if isinstance(data.get("provider"), dict) else data
    selection = provider.get("server_selection") if isinstance(provider, dict) else None
    if not isinstance(selection, dict):
        return None
    cities = selection.get("cities") or selection.get("Cities")
    if isinstance(cities, list) and cities:
        name = str(cities[0]).strip().lower()
        return NAME_TO_CITY.get(name) or name[:3]
    return None


def _gluetun_public_ip(base: str, fetch_ip_fn: Callable[[], str | None] | None) -> str | None:
    if fetch_ip_fn is not None:
        return fetch_ip_fn()
    status, body = _gluetun_request("GET", f"{base}/v1/publicip/ip")
    if status != 200:
        return fetch_public_ip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.strip() or None
    if isinstance(data, dict) and data.get("public_ip"):
        return str(data["public_ip"])
    if isinstance(data, dict) and data.get("ip"):
        return str(data["ip"])
    return body.strip() or None


def rotate_exit_gluetun(
    *,
    cities: tuple[str, ...] | None = None,
    log_path: Path | None = None,
    dry_run: bool = False,
    fetch_ip_fn: Callable[[], str | None] | None = None,
    request_fn: Callable[[str, str, dict | None], tuple[int, str]] | None = None,
    base_url: str | None = None,
) -> bool:
    """Select the next US city via Gluetun HTTP control. Returns True on success."""
    pool = cities if cities is not None else parse_cities()
    base = (base_url or gluetun_url()).rstrip("/")
    log = log_path or Path(catalog_goodreads(DEFAULT_LOG_NAME))
    requester = request_fn or (lambda method, url, payload=None: _gluetun_request(method, url, payload=payload))

    status_code, status_body = requester("GET", f"{base}/v1/vpn/status", None)
    settings_code, settings_body = requester("GET", f"{base}/v1/vpn/settings", None)
    before_city = _city_from_gluetun_settings(settings_body) if settings_code == 200 else None
    target = next_city(pool, before_city)
    city_name = GLUETUN_CITY_NAMES.get(target, target)
    connected = status_code == 200 and "running" in status_body.lower()
    ip_before = None if dry_run else _gluetun_public_ip(base, fetch_ip_fn)
    print(
        f"[mullvad_rotate] {before_city or 'unknown'} -> us {target} "
        f"(gluetun connected={connected}, dry_run={dry_run})"
    )
    if dry_run:
        _append_log(
            log,
            {
                "ts": _now_iso(),
                "ok": True,
                "dry_run": True,
                "backend": "gluetun",
                "from": before_city,
                "to": target,
            },
        )
        return True

    put_code, put_body = requester(
        "PUT",
        f"{base}/v1/vpn/settings",
        {"provider": {"server_selection": {"cities": [city_name]}}},
    )
    if put_code not in {200, 201, 204}:
        print(f"[mullvad_rotate] gluetun settings failed ({put_code}: {put_body[:200]})")
        _append_log(
            log,
            {
                "ts": _now_iso(),
                "ok": False,
                "backend": "gluetun",
                "from": before_city,
                "to": target,
                "error": put_body[:500],
            },
        )
        return False

    after_code, after_body = requester("GET", f"{base}/v1/vpn/status", None)
    after_settings_code, after_settings = requester("GET", f"{base}/v1/vpn/settings", None)
    after_city = _city_from_gluetun_settings(after_settings) if after_settings_code == 200 else None
    ip_after = _gluetun_public_ip(base, fetch_ip_fn)
    ok = after_code == 200 and "running" in after_body.lower()
    if after_city is not None and after_city != target:
        ok = False
    if ip_before and ip_after and ip_before == ip_after:
        print(f"[mullvad_rotate] public IP unchanged ({ip_after}) after {target}")
        ok = False
    record = {
        "ts": _now_iso(),
        "ok": ok,
        "backend": "gluetun",
        "from": before_city,
        "to": target,
        "ip_before": ip_before,
        "ip_after": ip_after,
        "connected": ok,
    }
    _append_log(log, record)
    if ok:
        print(f"[mullvad_rotate] connected us {target}")
    else:
        print(f"[mullvad_rotate] did not confirm {target} (status={after_body[:120]})")
    return ok


def rotate_exit(
    *,
    cities: tuple[str, ...] | None = None,
    country: str = DEFAULT_COUNTRY,
    log_path: Path | None = None,
    dry_run: bool = False,
    run_fn: RunFn | None = None,
    bin_path: str | None = None,
    fetch_ip_fn: Callable[[], str | None] | None = None,
) -> bool:
    """Disconnect, set the next US city, connect. Returns True on success."""
    if rotate_backend() == "gluetun":
        return rotate_exit_gluetun(
            cities=cities,
            log_path=log_path,
            dry_run=dry_run,
            fetch_ip_fn=fetch_ip_fn,
        )
    pool = cities if cities is not None else parse_cities()
    runner = run_fn or default_run_fn
    binary = bin_path or find_mullvad_bin()
    log = log_path or Path(catalog_goodreads(DEFAULT_LOG_NAME))
    if binary is None:
        print("[mullvad_rotate] mullvad CLI not found — skip")
        _append_log(log, {"ts": _now_iso(), "ok": False, "error": "missing_binary"})
        return False

    before = _run_status(binary, runner)
    target = next_city(pool, before.city)
    ip_before = fetch_ip_fn() if fetch_ip_fn else None
    print(
        f"[mullvad_rotate] {before.city or 'unknown'} -> {country} {target} "
        f"(connected={before.connected}, dry_run={dry_run})"
    )
    if dry_run:
        _append_log(
            log,
            {
                "ts": _now_iso(),
                "ok": True,
                "dry_run": True,
                "from": before.city,
                "to": target,
            },
        )
        return True

    steps = (
        [binary, "disconnect"],
        [binary, "relay", "set", "location", country, target],
        [binary, "connect", "--wait"],
    )
    for argv in steps:
        proc = runner(argv)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = err[-1] if err else f"exit {proc.returncode}"
            print(f"[mullvad_rotate] failed: {' '.join(argv)} ({tail})")
            _append_log(
                log,
                {
                    "ts": _now_iso(),
                    "ok": False,
                    "from": before.city,
                    "to": target,
                    "error": tail,
                    "cmd": argv[1:],
                },
            )
            return False

    after = _run_status(binary, runner)
    ip_after = fetch_ip_fn() if fetch_ip_fn else None
    ok = after.connected and (after.city is None or after.city == target)
    if ip_before and ip_after and ip_before == ip_after:
        print(f"[mullvad_rotate] public IP unchanged ({ip_after}) after {target}")
        ok = False
    record = {
        "ts": _now_iso(),
        "ok": ok,
        "from": before.city,
        "to": target,
        "hostname": after.hostname,
        "ip_before": ip_before,
        "ip_after": ip_after,
        "connected": after.connected,
    }
    _append_log(log, record)
    if ok:
        print(f"[mullvad_rotate] connected {country} {target}")
    else:
        print(f"[mullvad_rotate] did not confirm {target} (connected={after.connected}, city={after.city})")
    return ok


def maybe_rotate_exit(**kwargs) -> bool:
    """No-op unless `MULLVAD_ROTATE` is enabled. Returns True only after a real rotate."""
    if not rotate_enabled():
        print(f"[mullvad_rotate] skipped ({ENV_ROTATE} not set)")
        return False
    kwargs.setdefault("fetch_ip_fn", fetch_public_ip)
    return rotate_exit(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Log the next city without calling mullvad.")
    parser.add_argument(
        "--require-enabled",
        action="store_true",
        help="Honor MULLVAD_ROTATE (loop default). Without this flag, an explicit CLI run always rotates.",
    )
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.require_enabled and not rotate_enabled():
        print(f"[mullvad_rotate] skipped ({ENV_ROTATE} not set)")
        return 0
    ok = rotate_exit(
        dry_run=args.dry_run,
        log_path=args.log,
        fetch_ip_fn=None if args.dry_run else fetch_public_ip,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
