#!/usr/bin/env python3
"""Export the TO-MHT package into the external release repository."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

IGNORED_DIR_NAMES = {"__pycache__", "doc", "docs"}
IGNORED_FILE_NAMES = {".DS_Store"}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}
RELEASE_REPO_NAME = "stonesoup-tomht"
RELEASE_LOG_NAME = "RELEASE_HISTORY.md"


class ExportError(RuntimeError):
    """Raised for user-actionable export failures."""


def git_output(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise ExportError(f"git {' '.join(args)} failed in {repo}: {message}") from exc
    return result.stdout.strip()


def git_root(path: Path) -> Path:
    if not path.exists():
        raise ExportError(f"Path does not exist: {path}")
    return Path(git_output(path, "rev-parse", "--show-toplevel")).resolve()


def git_status(repo: Path) -> str:
    return git_output(repo, "status", "--porcelain", "--untracked-files=all")


def format_status(status: str) -> str:
    lines = status.splitlines()
    if len(lines) > 40:
        lines = [*lines[:40], f"... {len(lines) - 40} more line(s) omitted"]
    return "\n".join(f"  {line}" for line in lines)


def resolve_destination_repo(source_repo: Path) -> Path:
    candidate = source_repo.parent / RELEASE_REPO_NAME
    if not candidate.exists():
        raise ExportError(f"Could not find sibling release repo: {candidate}")
    return git_root(candidate)


def should_ignore_path(path: Path) -> bool:
    return (
        any(part in IGNORED_DIR_NAMES for part in path.parts[:-1])
        or path.name in IGNORED_DIR_NAMES
        or path.name in IGNORED_FILE_NAMES
        or path.suffix in IGNORED_FILE_SUFFIXES
    )


def copytree_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(name)
        if (
            name in IGNORED_DIR_NAMES
            or name in IGNORED_FILE_NAMES
            or path.suffix in IGNORED_FILE_SUFFIXES
        ):
            ignored.add(name)
    return ignored


def included_files(source_mht: Path) -> list[Path]:
    files = []
    for path in source_mht.rglob("*"):
        relative_path = path.relative_to(source_mht)
        if path.is_file() and not should_ignore_path(relative_path):
            files.append(relative_path)
    return sorted(files)


def release_log_path(dest_repo: Path) -> Path:
    release_log = dest_repo / RELEASE_LOG_NAME
    if not release_log.exists():
        raise ExportError(f"Could not find release log: {release_log}")
    return release_log


def release_entry(release_date: str, source_ref: str, notes: Sequence[str]) -> str:
    lines = [f"## {release_date}", ""]
    lines.extend(f"- {note}" for note in notes)
    lines.append(f"- Exported from `stone-soup-tracking` commit `{source_ref}`.")
    return "\n".join(lines)


def append_release_entry(
    release_log: Path,
    release_date: str,
    source_ref: str,
    notes: Sequence[str],
) -> None:
    existing = release_log.read_text(encoding="utf-8")
    updated = existing.rstrip()
    if updated:
        updated += "\n\n"
    updated += release_entry(release_date, source_ref, notes)
    updated += "\n"
    release_log.write_text(updated, encoding="utf-8")


def replace_release_mht(source_mht: Path, dest_mht: Path) -> None:
    if dest_mht.is_symlink():
        raise ExportError(f"Refusing to replace symlinked destination: {dest_mht}")
    if dest_mht.exists() and not dest_mht.is_dir():
        raise ExportError(f"Destination mht path is not a directory: {dest_mht}")
    if dest_mht.exists():
        shutil.rmtree(dest_mht)
    shutil.copytree(source_mht, dest_mht, ignore=copytree_ignore)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace stonesoup-tomht's mht/ snapshot and append a "
            "release-log entry pointing at the source commit."
        )
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Release-log date to use. Defaults to today.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Additional release-log bullet. May be passed multiple times.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty source tree and append -dirty to the source commit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned export without modifying the release repo.",
    )
    return parser.parse_args(argv)


def run_export(args: argparse.Namespace) -> None:
    source_repo = git_root(args.source_repo.expanduser().resolve())
    dest_repo = resolve_destination_repo(source_repo)
    if source_repo == dest_repo:
        raise ExportError("Source and destination repos resolve to the same path.")

    source_mht = source_repo / "mht"
    dest_mht = dest_repo / "mht"
    if not source_mht.is_dir():
        raise ExportError(f"Could not find source mht directory: {source_mht}")

    source_status = git_status(source_repo)
    if source_status and not args.allow_dirty:
        raise ExportError(
            "Source repo is dirty. Commit/stash changes first, or rerun with "
            f"--allow-dirty to record a dirty export.\n{format_status(source_status)}"
        )

    dest_status = git_status(dest_repo)
    if dest_status:
        raise ExportError(
            "Destination repo is dirty. Commit/stash changes before exporting.\n"
            f"{format_status(dest_status)}"
        )

    source_commit = git_output(source_repo, "rev-parse", "HEAD")
    source_ref = f"{source_commit}-dirty" if source_status else source_commit
    release_log = release_log_path(dest_repo)
    files = included_files(source_mht)

    if args.dry_run:
        print(f"Source repo:      {source_repo}")
        print(f"Destination repo: {dest_repo}")
        print(f"Source ref:       {source_ref}")
        print(f"Release date:     {args.date}")
        print(f"Release log:      {release_log}")
        print(f"Files to export:  {len(files)}")
        print(f"Would replace:    {dest_mht}")
        print("Dry run only; no files changed.")
        return

    replace_release_mht(source_mht, dest_mht)
    append_release_entry(release_log, args.date, source_ref, args.note)

    print(f"Exported {len(files)} file(s) to {dest_mht}")
    print(f"Updated {release_log}")
    print(f"Recorded source commit {source_ref}")
    print(f"Review with: git -C {dest_repo} status --short")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_export(parse_args(sys.argv[1:] if argv is None else argv))
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
