"""Command-line interface for kinetics-model-selector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import read_kinetics_data


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="kinetics-model-selector",
        description="Load kinetics concentration-time data for model selection workflows.",
    )
    parser.add_argument(
        "input_data",
        type=Path,
        help="Path to a CSV file with required columns: time, concentration.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory where outputs and intermediate artifacts will be written.",
    )
    return parser


def main() -> None:
    """Run the minimal CLI workflow."""
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_kinetics_data(args.input_data)

    print(
        f"Loaded {len(data)} rows from '{args.input_data}'. "
        f"Output directory: '{args.output_dir}'."
    )


if __name__ == "__main__":
    main()
