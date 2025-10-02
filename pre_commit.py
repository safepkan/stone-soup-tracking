#!/usr/bin/env python3
"""
Lint and format all Python files in the repository.
"""
import subprocess
import sys
from pathlib import Path
from typing import List


def _check_command(cmd: List[str]) -> bool:
    print("=" * 65)
    print(" ".join(cmd))
    print("=" * 65)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _get_python_files() -> List[str]:
    """Get all Python files in the repository."""
    all_py_files = list(Path(".").rglob("*.py"))
    # Exclude venv, external, and other common directories
    excluded_dirs = {"venv", "external"}
    py_files = [
        str(f)
        for f in all_py_files
        if not any(part.startswith(".") or part in excluded_dirs for part in f.parts)
    ]
    return py_files


def _check_python(pyfiles: List[str]) -> bool:
    """Run black, flake8, and mypy on Python files."""
    if not pyfiles:
        print("No Python files to check")
        return True

    print(f"Checking {len(pyfiles)} Python files:")
    print("\n".join(pyfiles))
    print()

    # Run black
    success = _check_command(["black"] + pyfiles)

    # Run flake8
    success = success and _check_command(["flake8"] + pyfiles)

    # Run mypy
    success = success and _check_command(
        ["mypy", "--install-types", "--non-interactive"] + pyfiles
    )

    return success


if __name__ == "__main__":
    pyfiles = _get_python_files()
    success = _check_python(pyfiles)
    sys.exit(0 if success else 1)
