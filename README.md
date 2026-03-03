# kinetics-model-selector

A lightweight starting point for selecting kinetic models from concentration-time datasets.

## Installation

### Standard install (with internet access)

```bash
python -m pip install -e .
```

Or install dependencies first:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

To include the optional Bayesian stack:

```bash
python -m pip install -e .[bayesian]
```

### Offline / restricted environment

If dependency downloads are blocked, you can still run the included test suite directly
from the repository checkout (uses the local compatibility shim):

```bash
pytest -q
```

## CLI Usage

Run the tested CLI workflow:

```bash
python -m kinetics_model_selector.cli --input tests/fixtures/pfo_clean.csv --outdir results/
```

This command writes:

- `results/summary.json`
- `results/mc_samples.csv`

### CLI arguments

- `--input`: Path to an input CSV file.
- `--outdir`: Output directory for generated artifacts.

## Expected Input Schema

The CLI in this repository expects CSV files with these columns:

- `t`: Time coordinate for each observation.
- `q`: Measured uptake/concentration response at the corresponding time.

Example:

```csv
t,q
0.5,0.7226
1.0,1.3929
1.5,2.0148
2.0,2.5918
```
