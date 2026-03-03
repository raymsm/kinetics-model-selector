"""CLI for model comparison and machine-readable output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from selection.compare import compare_models, comparison_to_rows


def _load_dataset(path: str, time_column: str, uptake_column: str) -> tuple[list[float], list[float]]:
    time: list[float] = []
    uptake: list[float] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if time_column not in reader.fieldnames or uptake_column not in reader.fieldnames:
            raise ValueError(
                f"Dataset must include columns '{time_column}' and '{uptake_column}'. Found: {reader.fieldnames}"
            )
        for row in reader:
            time.append(float(row[time_column]))
            uptake.append(float(row[uptake_column]))
    return time, uptake


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PFO/PSO/L-H model fits.")
    parser.add_argument("dataset", help="CSV file containing kinetics observations")
    parser.add_argument("--time-column", default="time", help="Column containing time values")
    parser.add_argument("--uptake-column", default="uptake", help="Column containing uptake values")
    parser.add_argument(
        "--criterion",
        choices=("r2", "aic", "bic"),
        default="r2",
        help="Primary model-selection criterion",
    )
    parser.add_argument("--json-out", help="Path to machine-readable JSON summary")
    parser.add_argument("--csv-out", help="Path to machine-readable CSV summary")
    parser.add_argument(
        "--skip-info-criteria",
        action="store_true",
        help="Skip AIC/BIC calculation and omit them from output",
    )
    parser.add_argument("--residual-dir", help="Directory to write residual plot assets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    time, uptake = _load_dataset(args.dataset, args.time_column, args.uptake_column)

    result = compare_models(
        time,
        uptake,
        criterion=args.criterion,
        include_info_criteria=not args.skip_info_criteria,
    )

    payload = result.to_dict()
    rows = comparison_to_rows(result)

    if args.json_out:
        _write_json(args.json_out, payload)

    if args.csv_out:
        _write_csv(args.csv_out, rows)

    if args.residual_dir:
        from plotting.residuals import save_residual_plots

        try:
            save_residual_plots(time, result, args.residual_dir)
        except RuntimeError as exc:
            print(f"Warning: {exc}")

    print(f"Best model: {result.best_model} ({result.criterion})")


if __name__ == "__main__":
    main()
