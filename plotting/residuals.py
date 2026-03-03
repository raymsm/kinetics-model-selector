"""Residual plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from selection.compare import ComparisonResult


def save_residual_plots(
    time: Iterable[float],
    comparison: ComparisonResult,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    """Generate per-model residual plots and save each in requested formats."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate residual plots") from exc

    t = [float(v) for v in time]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for model in ("PFO", "PSO", "L-H"):
        metric = comparison.metrics[model]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.scatter(t, metric.residuals, s=24)
        ax.set_xlabel("Time")
        ax.set_ylabel("Residual (observed - predicted)")
        ax.set_title(f"Residuals: {model}")
        ax.grid(alpha=0.3)

        for ext in formats:
            path = out / f"residuals_{model.replace('-', '').lower()}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    return written
