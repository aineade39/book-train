#!/usr/bin/env python3
"""Orchestrate OL download → ETL → Swift catalog-build for catalog profiles.

See docs/BOOK_CATALOG.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.ol_common import load_profiles  # noqa: E402
from tools.derived_meta import git_commit_short  # noqa: E402
from tools.paths import catalog_dir, catalog_intermediate, catalog_raw_ol  # noqa: E402

PROFILES_PATH = _REPO / "tools" / "catalog" / "profiles.yaml"
BOOK_ID_IOS_RESOURCES = _REPO.parent / "book-id-ios" / "Sources" / "BookID" / "Resources" / "ios_en.sqlite"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=cwd or _REPO)


def intermediate_fresh(intermediate: Path, raw: Path) -> bool:
    manifest = intermediate / "manifest.json"
    if not manifest.exists():
        return False
    for name in (
        "ol_dump_editions_latest.txt.gz",
        "ol_dump_works_latest.txt.gz",
        "ol_dump_authors_latest.txt.gz",
    ):
        if not (raw / name).exists():
            return False
    return (intermediate / "works.jsonl.gz").exists()


def write_sidecar(sqlite_path: Path, profile_name: str, profile: dict, work_count: int) -> None:
    sidecar = {
        "profile": profile_name,
        "works": work_count,
        "bytes": sqlite_path.stat().st_size if sqlite_path.exists() else 0,
        "git_commit": git_commit_short(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filters": {
            "languages": profile.get("languages", []),
            "max_works": profile.get("max_works"),
            "min_editions": profile.get("min_editions", 1),
        },
    }
    sqlite_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def swift_catalog_build(args: list[str]) -> None:
    run(["swift", "run", "-c", "release", "catalog-build", *args])


def build_profile(
    profile_name: str,
    profile: dict,
    *,
    intermediate: Path,
    subset_from: Path | None,
    reuse_intermediate: bool,
) -> Path:
    out = catalog_dir(f"{profile_name}.sqlite")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["--output", str(out)]
    langs = profile.get("languages") or []
    if langs:
        cmd.extend(["--languages", *langs])
    max_works = profile.get("max_works")
    if max_works is not None:
        cmd.extend(["--max-works", str(max_works)])
    min_editions = profile.get("min_editions", 1)
    cmd.extend(["--min-editions", str(min_editions)])

    if subset_from:
        cmd.extend(["--subset-from", str(subset_from)])
    else:
        cmd.extend(["--intermediate", str(intermediate)])

    swift_catalog_build(cmd)

    catalog = None
    try:
        import sqlite3

        conn = sqlite3.connect(out)
        work_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        conn.close()
    except OSError:
        work_count = 0
    write_sidecar(out, profile_name, profile, work_count)
    print(f"Built {out} ({work_count} works)", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", help="Profile name (repeatable)")
    parser.add_argument("--all", action="store_true", help="Build all profiles")
    parser.add_argument("--reuse-intermediate", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--subset-from", type=Path, default=None, help="Source full.sqlite for fast subset rebuild")
    parser.add_argument("--install-ios", action="store_true", help="Copy ios_en.sqlite into book-id-ios Resources")
    parser.add_argument("--fixture", type=Path, default=None, help="Use Tests/fixtures/ol-mini instead of OL dumps")
    args = parser.parse_args()

    profiles = load_profiles(PROFILES_PATH)
    # Profiles with a `custom_build_script` (e.g. `ios_en_shelf`) need SQL
    # mutation this generic subset-from path can't do (a per-work language
    # filter on a subset build, Goodreads re-ranking/gap-fill — see
    # tools/catalog/profiles.yaml's comment) and must be built by that
    # script directly, never through here.
    custom_build_profiles = {n for n, p in profiles.items() if p.get("custom_build_script")}
    if args.all:
        names = ["full"] + [n for n in profiles if n != "full" and n not in custom_build_profiles]
    elif args.profile:
        names = args.profile
    else:
        names = ["ios_en"]

    for name in names:
        script = profiles.get(name, {}).get("custom_build_script")
        if script:
            print(
                f"Profile '{name}' is built by `python {script}`, not build_book_catalog.py "
                f"(see tools/catalog/profiles.yaml).",
                file=sys.stderr,
            )
            return 1

    raw = args.fixture or catalog_raw_ol()
    intermediate = catalog_intermediate()

    if args.fixture:
        if not args.reuse_intermediate or not (intermediate / "works.jsonl.gz").exists():
            run(
            [
                sys.executable,
                str(_REPO / "tools" / "catalog" / "process_ol.py"),
                "--raw-dir",
                str(args.fixture),
                "--editions",
                str(args.fixture / "editions.jsonl"),
                "--works",
                str(args.fixture / "works.jsonl"),
                "--authors",
                str(args.fixture / "authors.jsonl"),
                "--out-dir",
                str(intermediate),
                "--min-editions",
                "1",
            ]
            )
    else:
        if not args.skip_download and not args.reuse_intermediate:
            run([sys.executable, str(_REPO / "tools" / "catalog" / "download_ol_dumps.py")])
        if not args.reuse_intermediate or not intermediate_fresh(intermediate, raw):
            run([sys.executable, str(_REPO / "tools" / "catalog" / "process_ol.py")])

    subset_from = args.subset_from
    built_ios: Path | None = None
    for name in names:
        if name not in profiles:
            print(f"Unknown profile: {name}", file=sys.stderr)
            return 1
        profile_subset = None
        if name != "full":
            if subset_from is not None:
                profile_subset = subset_from
            else:
                full_path = catalog_dir("full.sqlite")
                if full_path.exists():
                    profile_subset = full_path
        out = build_profile(
            name,
            profiles[name],
            intermediate=intermediate,
            subset_from=profile_subset,
            reuse_intermediate=args.reuse_intermediate,
        )
        if name == "ios_en":
            built_ios = out

    if args.install_ios and built_ios:
        BOOK_ID_IOS_RESOURCES.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(built_ios, BOOK_ID_IOS_RESOURCES)
        print(f"Installed {built_ios} → {BOOK_ID_IOS_RESOURCES}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
