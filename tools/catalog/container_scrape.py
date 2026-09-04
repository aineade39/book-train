"""Host-side helpers for the ISBN scrape container (prepare / start gates)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tools.paths import book_spines_data, catalog_goodreads

ITEM_ID = "ml-goodreads-scrape"
MARKER_NAME = "container_migration.json"
CAFFEINATE_PID_NAME = "run_book_show_api_caffeinate.pid"
NAS_ROOT = Path("/Volumes/home")
DISK_CLEANUP_ROOT = Path.home() / "dev" / "disk-cleanup"
MANIFEST_ROOT = Path.home() / ".local" / "share" / "disk-cleanup" / "manifests"
COMPOSE_DIR = Path(__file__).resolve().parents[2] / "tools" / "scrape-container"
JSONL_NAME = "book_show_api.jsonl"


def marker_path() -> Path:
    return Path(catalog_goodreads(MARKER_NAME))


def jsonl_path() -> Path:
    return Path(catalog_goodreads(JSONL_NAME))


def caffeinate_pid_path() -> Path:
    return Path(catalog_goodreads(CAFFEINATE_PID_NAME))


def host_data_root() -> Path:
    return book_spines_data().resolve()


def jsonl_stat(path: Path | None = None) -> dict[str, int]:
    target = path or jsonl_path()
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty JSONL missing: {target}")
    with target.open(encoding="utf-8", errors="replace") as handle:
        lines = sum(1 for _ in handle)
    return {"bytes": target.stat().st_size, "lines": lines}


def compose_bind_source() -> Path:
    raw = os.environ.get("BOOK_SPINES_DATA") or str(Path.home() / "ml" / "book-spines")
    return Path(raw).expanduser().resolve()


def compose_bind_matches_host() -> bool:
    return compose_bind_source() == host_data_root()


def latest_manifest(item_id: str = ITEM_ID) -> dict | None:
    directory = MANIFEST_ROOT / item_id
    if not directory.is_dir():
        return None
    runs = sorted(directory.glob("*.json"))
    if not runs:
        return None
    return json.loads(runs[-1].read_text(encoding="utf-8"))


def backup_verified(item_id: str = ITEM_ID) -> tuple[bool, str]:
    if not NAS_ROOT.is_dir():
        return False, f"NAS not mounted at {NAS_ROOT}"
    manifest = latest_manifest(item_id)
    if manifest is None:
        return False, f"no disk-cleanup manifest for {item_id}"
    nas = manifest.get("nas") or {}
    if nas.get("status") != "verified":
        return False, f"{item_id} NAS stage is {nas.get('status')!r}, not verified"
    if manifest.get("status") != "verified":
        return False, f"{item_id} GDrive stage is {manifest.get('status')!r}, not verified"
    source = Path(manifest.get("source") or "")
    expected = Path(catalog_goodreads()).resolve()
    if source.resolve() != expected:
        return False, f"manifest source {source} != {expected}"
    return True, manifest.get("run_id") or ""


def write_marker(*, run_id: str, stat: dict[str, int]) -> Path:
    path = marker_path()
    payload = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "run_id": run_id,
        "item_id": ITEM_ID,
        "host_data_root": str(host_data_root()),
        "jsonl": str(jsonl_path().resolve()),
        "bytes": stat["bytes"],
        "lines": stat["lines"],
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_marker() -> dict | None:
    path = marker_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def ensure_harness_bind_files(harness_root: Path) -> None:
    chunked = harness_root / "sites" / "goodreads" / "profiles" / "book_show_api_chunked.yaml"
    next_build = harness_root / "sites" / "goodreads" / "har" / "next_build.yaml"
    chunked.parent.mkdir(parents=True, exist_ok=True)
    next_build.parent.mkdir(parents=True, exist_ok=True)
    if not chunked.exists():
        chunked.write_text("# generated at scrape time\n", encoding="utf-8")
    if not next_build.exists():
        next_build.write_text("{}\n", encoding="utf-8")
