"""Figure stage: CSV in, PDF out.

Each task declares its CSV as a plain ``Path`` default argument (a dependency)
and its PDF via ``Product``. pytask reads the annotations and orders the graph, so
these run after the simulation tasks without any manual sequencing - and rerun
when the CSV hash changes.

``plotting.py`` and ``config.py`` are declared as dependencies explicitly.
pytask hashes a task module's own source and its declared files, but *not* the
first-party modules it imports - so without these lines, editing a figure's
colours or layout would leave the stale PDF on disk and pytask would report
"skipped because unchanged".

Export goes through kaleido 0.2.1, pinned because it bundles its own Chromium;
kaleido >= 1 shells out to system Chrome, which no lockfile can pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from stpetersburg.config import DATA_PROCESSED, FIGURES, N_PATHS_SHOWN, SRC
from stpetersburg.plotting import ensemble_convergence, time_paths


def task_figure_ensemble_convergence(
    means_csv: Path = DATA_PROCESSED / "ensemble_means.csv",
    plotting_module: Path = SRC / "plotting.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = FIGURES / "ensemble_convergence.pdf",
) -> None:
    """Sample mean against sample size: the non-convergence figure."""
    means = pd.read_csv(means_csv)
    fig = ensemble_convergence(means, n_paths_shown=N_PATHS_SHOWN)
    fig.write_image(produces)


def task_figure_time_paths(
    paths_csv: Path = DATA_PROCESSED / "time_paths.csv",
    plotting_module: Path = SRC / "plotting.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = FIGURES / "time_paths.pdf",
) -> None:
    """Wealth of a single player over time at three ticket prices."""
    paths = pd.read_csv(paths_csv)
    fig = time_paths(paths)
    fig.write_image(produces)
