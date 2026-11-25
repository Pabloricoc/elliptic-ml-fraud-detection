"""Command line access to validation and local exploratory reports."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .analysis import summarize
from .ingestion import load_dataset
from .report import write_report
from .schema import DatasetError


def main(argv: Sequence[str] | None = None) -> int:
    """Run structural checks and optionally write aggregate statistics and a plot."""
    parser = argparse.ArgumentParser(
        description="Inspect the original Elliptic transaction dataset."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="validate tables and report quality warnings"
    )
    analyze = commands.add_parser(
        "analyze", help="validate and write a local EDA report"
    )
    for command in (validate, analyze):
        command.add_argument(
            "--data-dir",
            type=Path,
            default=Path("data/raw"),
            help="original CSV directory (default: data/raw)",
        )
    analyze.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eda"),
        help="parent directory for completed reports",
    )
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.data_dir)
        if args.command == "analyze":
            if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()):
                raise DatasetError(
                    "Choose a report directory outside the raw input directory."
                )
            destination, summary = write_report(dataset, args.output_dir)
            print(f"Report written to {destination}")
        else:
            summary, _ = summarize(dataset)
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        f"Validated {len(dataset.transactions)} transactions and {len(dataset.edges)} directed edges."
    )
    for warning in summary.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0
