"""Table stage: CSV in, LaTeX fragment out.

The .tex files are fragments (``tabular`` only, no ``table`` float and no
caption), so the paper controls placement and captioning while the numbers stay
generated. Never hand-edit these - edit the task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from stpetersburg.config import DATA_PROCESSED, N_CHECKPOINTS, TABLES


def task_table_ensemble_summary(
    means_csv: Path = DATA_PROCESSED / "ensemble_means.csv",
    produces: Annotated[Path, Product] = TABLES / "ensemble_summary.tex",
) -> None:
    """Sampling distribution of the mean payoff at decade checkpoints.

    Reporting median and quantiles alongside the mean is the point: where the mean
    exceeds its own 95th percentile, a single simulation run is almost certain to
    land below the expected value.
    """
    means = pd.read_csv(means_csv)
    at_checkpoints = means.loc[means["n"].isin(N_CHECKPOINTS)]
    summary = (
        at_checkpoints.groupby("n")["mean_payoff"]
        .agg(
            median="median",
            mean="mean",
            q05=lambda s: s.quantile(0.05),
            q95=lambda s: s.quantile(0.95),
            maximum="max",
        )
        .reset_index()
    )
    summary.to_latex(
        produces,
        index=False,
        float_format="%.1f",
        column_format="rrrrrr",
        header=["$n$", "median", "mean", "5\\%", "95\\%", "max"],
        escape=False,
    )


def task_table_time_summary(
    summary_csv: Path = DATA_PROCESSED / "time_summary.csv",
    produces: Annotated[Path, Product] = TABLES / "time_summary.tex",
) -> None:
    """Ruin frequency and typical outcome per ticket price."""
    summary = pd.read_csv(summary_csv)
    summary.to_latex(
        produces,
        index=False,
        float_format="%.3f",
        column_format="rrrr",
        header=["cost", "median final wealth", "ruin fraction", "gain per round"],
        escape=False,
    )
