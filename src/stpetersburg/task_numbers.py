"""Export the numbers quoted in the paper as LaTeX macros.

The paper never hard-codes a simulated figure. This task writes one
``\\newcommand`` per quoted quantity; ``main.tex`` inputs the file in its
preamble and the prose refers to ``\\simMedianEnd`` rather than to "18.8". Change
the seed or the sample size and the sentences update on the next build, so the
text cannot drift out of step with the simulation behind it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from stpetersburg.config import (
    DATA_PROCESSED,
    N_CHECKPOINTS,
    N_REPLICATIONS,
    SRC,
    TABLES,
)


def _thousands(value: float) -> str:
    """LaTeX-safe thousands separator: ``100{,}000`` keeps math-mode spacing right."""
    return f"{round(value):,}".replace(",", "{,}")


def task_simulation_numbers(
    means_csv: Path = DATA_PROCESSED / "ensemble_means.csv",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = TABLES / "simulation_numbers.tex",
) -> None:
    """Write the quantities the simulation section quotes."""
    tidy = pd.read_csv(means_csv)
    # The endpoint comes from the data, not from a constant. N_CHECKPOINTS and
    # N_MAX are separate knobs, so reading the last checkpoint here would quote
    # statistics from the middle of the grid whenever N_MAX is raised on its own.
    n_start, n_end = N_CHECKPOINTS[0], int(tidy["n"].max())
    start = tidy.loc[tidy["n"] == n_start, "mean_payoff"]
    end = tidy.loc[tidy["n"] == n_end, "mean_payoff"]

    # The headline comparison: at which sample sizes does the mean of the
    # estimator exceed its own 95th percentile? Computed, not asserted.
    by_n = tidy.groupby("n")["mean_payoff"]
    crossings = by_n.mean() > by_n.quantile(0.95)

    macros = {
        "simReps": f"{N_REPLICATIONS}",
        "simNStart": _thousands(n_start),
        "simNEnd": _thousands(n_end),
        "simMedianStart": f"{start.median():.1f}",
        "simMedianEnd": f"{end.median():.1f}",
        "simMeanEnd": f"{end.mean():.1f}",
        "simQNinetyFiveEnd": f"{end.quantile(0.95):.1f}",
        "simMaxEnd": _thousands(end.max()),
        "simMedianRatio": f"{end.median() / start.median():.1f}",
        "simSpan": _thousands(n_end / n_start),
        "simCrossings": f"{int(crossings.sum())}",
        "simGridPoints": f"{tidy['n'].nunique()}",
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    produces.write_text("\n".join(lines) + "\n")
