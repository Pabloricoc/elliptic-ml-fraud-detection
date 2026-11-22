"""Inspect the local directory layout of the original Elliptic release."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

EXPECTED_FILES = (
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_features.csv",
)


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
        missing = [name for name in EXPECTED_FILES if not (directory / name).is_file()]
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
