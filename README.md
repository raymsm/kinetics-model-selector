# kinetics-model-selector

A lightweight starting point for selecting kinetic models from concentration-time datasets.

## Installation

```bash
pip install -e .
```

Or install runtime dependencies directly:

```bash
pip install -r requirements.txt
```

To include the optional Bayesian stack:

```bash
pip install -e .[bayesian]
```

## CLI Usage

```bash
kinetics-model-selector path/to/data.csv --output-dir results/
```

Arguments:

- `input_data`: Path to the input CSV file.
- `--output-dir` / `-o`: Output directory for generated artifacts.

## Expected Input Schema

Input CSV files must include these columns:

- `time`: Time coordinate for each observation.
- `concentration`: Measured concentration at the corresponding time.

Example:

```csv
time,concentration
0,1.00
1,0.81
2,0.67
3,0.55
```
