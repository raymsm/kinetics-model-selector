from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .models import langmuir_hinshelwood, pfo, pso
from .selection import fit_models, select_best_model
from .uncertainty import monte_carlo_parameter_samples


def _read_csv(path: Path) -> tuple[list[float], list[float]]:
    t_vals: list[float] = []
    q_vals: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_vals.append(float(row["t"]))
            q_vals.append(float(row["q"]))
    return t_vals, q_vals


def _write_samples(path: Path, param_names: list[str], samples: list[list[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(param_names)
        writer.writerows(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t, q = _read_csv(input_path)
    all_fits = fit_models(t, q)
    best_name, best_fit = select_best_model(t, q)

    model_map = {
        "pfo": (pfo.fit, ["qe", "k1"]),
        "pso": (pso.fit, ["qe", "k2"]),
        "langmuir_hinshelwood": (langmuir_hinshelwood.fit, ["k", "K"]),
    }
    fit_fn, param_names = model_map[best_name]
    samples, intervals = monte_carlo_parameter_samples(t, q, fit_fn, param_names, n_iter=25)

    summary = {
        "best_model": best_name,
        "best_fit": best_fit,
        "all_fits": all_fits,
        "param_names": param_names,
        "intervals": intervals,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_samples(outdir / "mc_samples.csv", param_names, samples)


if __name__ == "__main__":
    main()
