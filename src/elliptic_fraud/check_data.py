"""Inspect the local directory layout of the original Elliptic release."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .schema import EXPECTED_FILES, missing_files


def main(argv: Sequence[str] | None = None) -> int:
    """Report missing regular files and return an exit status for shell use."""
    parser = argparse.ArgumentParser(
        description="Check for the three original Elliptic CSV filenames."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("data", "raw"),
        help="directory containing the CSV files (default: data/raw)",
    )
    directory = parser.parse_args(argv).directory

    try:
        if not directory.is_dir():
            print(f"Expected a directory at {directory}.", file=sys.stderr)
            return 1
        missing = missing_files(directory)
    except OSError as error:
        print(f"Could not inspect {directory}: {error}", file=sys.stderr)
        return 1

    if missing:
        print(
            f"Missing regular files in {directory}:",
            *missing,
            sep="\n",
            file=sys.stderr,
        )
        return 1

    print(f"Found all {len(EXPECTED_FILES)} expected filenames in {directory}.")
    return 0
