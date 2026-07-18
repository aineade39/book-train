"""Fetch raw dataset archives from Drive, unpack to temp, process from a directory.

Policy (see DATA.md): Drive keeps zip + SOURCE.md; local raw/ keeps SOURCE.md only.
Prep scripts always consume an unpacked directory (never zip members in place).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from paths import book_spines_data, raw_dir

_FIELD_RE = re.compile(r"\|\s*\*\*([^*]+)\*\*\s*\|\s*(.*?)\s*\|")


def tmp_root() -> Path:
    root = book_spines_data() / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_md_path(dataset_id: str) -> Path:
    return raw_dir(dataset_id, "SOURCE.md")


def load_source(dataset_id: str) -> dict[str, str]:
    path = source_md_path(dataset_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Each raw dataset needs a SOURCE.md sidecar (see DATA.md)."
        )
    meta: dict[str, str] = {"id": dataset_id}
    for key, raw_val in _FIELD_RE.findall(path.read_text(encoding="utf-8")):
        val = raw_val.strip()
        # Strip markdown links [text](url) -> text, and backticks.
        link = re.fullmatch(r"\[([^\]]+)\]\([^)]+\)", val)
        if link:
            val = link.group(1)
        val = val.strip("`").strip()
        meta[key.strip().lower().replace(" ", "_")] = val
    if "archive" not in meta:
        raise ValueError(f"{path}: SOURCE.md must define **archive** (zip filename).")
    if "drive_path" not in meta:
        raise ValueError(f"{path}: SOURCE.md must define **drive_path** (rclone remote path).")
    return meta


def _rclone_copy_file(remote_file: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", remote_file, str(dest_dir), "--progress"]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    name = Path(remote_file.rstrip("/")).name
    out = dest_dir / name
    if not out.is_file():
        raise FileNotFoundError(f"rclone copy finished but missing {out}")
    return out


def _unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Unpacking {archive.name} -> {dest}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def resolve_content_root(unpack_dir: Path, content_root: str | None) -> Path:
    if content_root:
        root = unpack_dir / content_root
        if not root.is_dir():
            raise FileNotFoundError(
                f"content_root {content_root!r} not found under {unpack_dir}. "
                f"Entries: {[p.name for p in unpack_dir.iterdir()][:20]}"
            )
        return root
    return unpack_dir


@contextmanager
def unpacked_raw(
    dataset_id: str,
    *,
    keep_tmp: bool = False,
    override_root: Path | None = None,
) -> Iterator[Path]:
    """Yield an unpacked dataset directory.

    If ``override_root`` is set, yield it directly (no fetch).
    Otherwise: read SOURCE.md → use local archive if present else rclone from
    Drive → unzip under ``$BOOK_SPINES_DATA/tmp/`` → yield ``content_root``.
    """
    if override_root is not None:
        root = override_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"override root not a directory: {root}")
        yield root
        return

    meta = load_source(dataset_id)
    archive_name = meta["archive"]
    drive_path = meta["drive_path"].rstrip("/")
    content_root = meta.get("content_root") or None
    if content_root in {"", ".", "-"}:
        content_root = None

    tmp = Path(tempfile.mkdtemp(prefix=f"{dataset_id}-", dir=str(tmp_root())))
    local_archive = raw_dir(dataset_id, archive_name)
    try:
        if local_archive.is_file():
            print(f"Using local archive {local_archive}", flush=True)
            archive_path = local_archive
        else:
            remote = f"{drive_path}/{archive_name}"
            archive_path = _rclone_copy_file(remote, tmp)

        unpack_dir = tmp / "unpack"
        _unzip(archive_path, unpack_dir)
        yield resolve_content_root(unpack_dir, content_root)
    finally:
        if keep_tmp:
            print(f"Keeping temp dir {tmp}", flush=True)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def sync_source_md(dataset_id: str) -> None:
    """Copy local SOURCE.md to Drive beside the archive."""
    meta = load_source(dataset_id)
    local = source_md_path(dataset_id)
    remote_dir = meta["drive_path"].rstrip("/")
    cmd = ["rclone", "copy", str(local), remote_dir, "--include", "SOURCE.md"]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
