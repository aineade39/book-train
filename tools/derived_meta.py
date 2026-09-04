"""Write provenance sidecars for derived datasets (see DATA.md)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except OSError:
        pass
    return None


def write_derived_source(
    out_root: Path,
    *,
    derived_id: str,
    title: str,
    sources: list[str],
    script: str,
    flags: dict[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    """Write ``SOURCE.md`` next to a derived dataset tree."""
    out_root.mkdir(parents=True, exist_ok=True)
    flags = flags or {}
    flag_lines = "\n".join(f"| `{k}` | `{v}` |" for k, v in flags.items()) or "| _(none)_ | |"
    sources_md = ", ".join(f"`{s}`" for s in sources)
    notes_line = notes or ""
    body = f"""# {derived_id}

Rebuildable derived dataset. Raw dumps stay under `raw/<id>/`; this tree is
produced by the script below and can be deleted/regenerated.

| Field | Value |
|---|---|
| **id** | `{derived_id}` |
| **title** | {title} |
| **sources** | {sources_md} |
| **script** | `{script}` |
| **git_commit** | `{git_commit_short() or "unknown"}` |
| **built_at** | `{datetime.now(timezone.utc).isoformat(timespec="seconds")}` |

## Flags

| Flag | Value |
|---|---|
{flag_lines}

{notes_line}
"""
    path = out_root / "SOURCE.md"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def dataset_tag_from_dir(data_yaml: Path) -> str:
    """Short run-name prefix from a derived folder name."""
    name = data_yaml.parent.name
    for suffix in ("_yolo-obb", "_createml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.startswith("yolo-obb-"):
        return name[len("yolo-obb-") :]
    return name
