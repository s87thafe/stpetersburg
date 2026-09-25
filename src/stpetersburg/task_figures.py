"""Figure stage: CSV in, PDF out.

The CSV is a plain ``Path`` default argument (a dependency) and the PDF is
annotated ``Product``, so pytask orders this after the simulation without manual
sequencing. ``plotting.py`` and ``config.py`` are declared for the same reason as
in ``task_simulation.py``: pytask does not follow first-party imports, so a
change to the figure's design would otherwise leave a stale PDF in place.

Export goes through kaleido 0.2.1, pinned because it bundles its own Chromium.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from stpetersburg.config import DATA_PROCESSED, FIGURES, SRC
from stpetersburg.plotting import plot_ensemble_means


def task_figure_ensemble_means(
    means_csv: Path = DATA_PROCESSED / "ensemble_means.csv",
    plotting_module: Path = SRC / "plotting.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = FIGURES / "ensemble_means.pdf",
) -> None:
    """Quantile fan of the sample mean payoff against sample size."""
    tidy = pd.read_csv(means_csv)
    # Back to the (replication, n) matrix the plotting function expects.
    wide = tidy.pivot(index="replication", columns="n", values="mean_payoff")
    fig = plot_ensemble_means(wide.columns.to_numpy(), wide.to_numpy())
    fig.write_image(produces)
