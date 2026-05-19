#!/usr/bin/env python3
"""Install a stonesoup-tomht snapshot into a target folder."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

DEFAULT_TARGET = Path(
    "/home/eatpadn/isac/radiant-isac-coeur/apps/radiant_isac/"
    "radiant_isac/sandbox/fusion/mht"
)
IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "venv",
    "venv.bak",
}
IGNORED_NAME_PATTERNS = ("*.egg-info", ".venv*")
IGNORED_FILE_NAMES = {".DS_Store"}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}


class InstallError(RuntimeError):
    """Raised for user-actionable install failures."""


def is_preserved_target_entry(path: Path) -> bool:
    return path.name == "venv" or path.name.startswith(".venv")


def is_ignored_source_path(path: Path, is_dir: bool) -> bool:
    if path.name in IGNORED_FILE_NAMES or path.suffix in IGNORED_FILE_SUFFIXES:
        return True
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in IGNORED_NAME_PATTERNS):
        return True
    return is_dir and path.name in IGNORED_DIR_NAMES


def copytree_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(directory) / name
        if is_ignored_source_path(path, path.is_dir()):
            ignored.add(name)
    return ignored


def is_snapshot_root(path: Path) -> bool:
    return (path / "mht").is_dir() and (path / "RELEASE_HISTORY.md").is_file()


def find_snapshot_root(path: Path) -> Path:
    if is_snapshot_root(path):
        return path

    candidates = [
        child
        for child in sorted(path.iterdir())
        if child.is_dir() and is_snapshot_root(child)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise InstallError(
            f"Could not find a stonesoup-tomht snapshot root under {path}; "
            "expected mht/ and RELEASE_HISTORY.md."
        )
    candidate_list = "\n".join(f"  {candidate}" for candidate in candidates)
    raise InstallError(
        f"Found multiple possible snapshot roots under {path}:\n{candidate_list}"
    )


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise InstallError(
                    f"Zip member escapes extraction directory: {member.filename}"
                )
        archive.extractall(destination)


def source_root_from_input(source: Path, temp_dir: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_dir():
        return find_snapshot_root(source)
    if source.is_file():
        if not zipfile.is_zipfile(source):
            raise InstallError(f"Input file is not a zip archive: {source}")
        safe_extract_zip(source, temp_dir)
        return find_snapshot_root(temp_dir)
    raise InstallError(f"Input path does not exist: {source}")


def included_source_files(source_root: Path) -> list[Path]:
    files = []
    for path in source_root.rglob("*"):
        relative_path = path.relative_to(source_root)
        ignored = any(
            is_ignored_source_path(source_root / part, True)
            for part in relative_path.parts[:-1]
        )
        if ignored:
            continue
        if path.is_file() and not is_ignored_source_path(relative_path, False):
            files.append(relative_path)
    return sorted(files)


def target_entries(target: Path) -> tuple[list[Path], list[Path]]:
    if not target.exists():
        return [], []

    removable = []
    preserved = []
    for entry in sorted(target.iterdir()):
        if is_preserved_target_entry(entry):
            preserved.append(entry)
        else:
            removable.append(entry)
    return removable, preserved


def assert_safe_target(target: Path) -> None:
    if target == target.parent:
        raise InstallError(f"Refusing to use filesystem root as target: {target}")
    if len(target.parts) <= 2:
        raise InstallError(f"Refusing suspiciously broad target path: {target}")
    if target == Path.home().resolve():
        raise InstallError(f"Refusing to use home directory as target: {target}")
    if target.exists() and not target.is_dir():
        raise InstallError(f"Target exists but is not a directory: {target}")


def assert_no_source_target_overlap(source_root: Path, target: Path) -> None:
    if source_root == target:
        raise InstallError("Source snapshot and target folder are the same path.")
    if source_root in target.parents:
        raise InstallError(f"Target folder is inside the source snapshot: {target}")
    if target in source_root.parents:
        raise InstallError(
            f"Source snapshot is inside the target folder: {source_root}"
        )


def remove_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_entry(source: Path, destination: Path) -> None:
    if destination.exists():
        raise InstallError(
            f"Destination entry already exists after cleanup: {destination}"
        )
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, destination, ignore=copytree_ignore, symlinks=True)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def replace_target_contents(source_root: Path, target: Path) -> tuple[int, list[Path]]:
    target.mkdir(parents=True, exist_ok=True)
    removable, preserved = target_entries(target)
    for entry in removable:
        remove_entry(entry)

    copied = 0
    for entry in sorted(source_root.iterdir()):
        if is_ignored_source_path(entry.relative_to(source_root), entry.is_dir()):
            continue
        copy_entry(entry, target / entry.name)
        copied += 1

    return copied, preserved


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a stonesoup-tomht GitHub zip or local snapshot folder into "
            "a target folder."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a stonesoup-tomht zip file or already-unpacked snapshot folder.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Target folder to replace. Defaults to {DEFAULT_TARGET}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned replacement without modifying the target folder.",
    )
    return parser.parse_args(argv)


def run_install(args: argparse.Namespace) -> None:
    target = args.target.expanduser().resolve()
    assert_safe_target(target)

    with tempfile.TemporaryDirectory(prefix="stonesoup-tomht-") as temp_name:
        source_root = source_root_from_input(args.source, Path(temp_name))
        assert_no_source_target_overlap(source_root, target)
        source_files = included_source_files(source_root)
        removable, preserved = target_entries(target)

        if args.dry_run:
            print(f"Source root:       {source_root}")
            print(f"Target:            {target}")
            print(f"Files to install:  {len(source_files)}")
            print(f"Entries to remove: {len(removable)}")
            print(f"Entries preserved: {len(preserved)}")
            for path in preserved:
                print(f"  preserve {path.name}")
            print("Dry run only; no files changed.")
            return

        copied_entries, preserved_entries = replace_target_contents(source_root, target)

    print(f"Installed snapshot from {source_root}")
    print(f"Replaced target: {target}")
    print(f"Copied {copied_entries} top-level entries")
    if preserved_entries:
        names = ", ".join(path.name for path in preserved_entries)
        print(f"Preserved target entries: {names}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_install(parse_args(sys.argv[1:] if argv is None else argv))
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
