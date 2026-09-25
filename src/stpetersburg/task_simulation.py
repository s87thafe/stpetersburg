"""Simulation stage: draw everything once, write a tidy CSV.

pytask discovers ``task_*`` functions in ``task_*.py`` modules. The output is
annotated ``Annotated[Path, Product]``, which is how pytask learns the dependency
graph: it hashes the product and the task's own source and reruns only what
changed.

``simulation.py`` and ``config.py`` are declared as dependencies explicitly.
pytask hashes a task module's own source and its declared files, but does *not*
follow first-party imports - so without those arguments, changing the seed or the
payoff rule would leave a stale CSV on disk and pytask would report "skipped
because unchanged".
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from stpetersburg.config import (
    DATA_PROCESSED,
    N_CHECKPOINTS,
    N_MAX,
    N_REPLICATIONS,
    SEED,
    SRC,
)
from stpetersburg.simulation import convergence_grid, make_rng, mc_ensemble_means


def task_ensemble_means(
    simulation_module: Path = SRC / "simulation.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = DATA_PROCESSED / "ensemble_means.csv",
) -> None:
    """Sample mean payoff against sample size, repeated N_REPLICATIONS times."""
    rng = make_rng(SEED)
    n_grid = convergence_grid(N_MAX, N_CHECKPOINTS)
    means = mc_ensemble_means(n_grid, N_REPLICATIONS, rng)
    # Tidy, not wide: one row per (replication, n) survives adding parameters
    # later and reads straight into a plotting or summary step.
    tidy = pd.DataFrame(means, columns=n_grid).stack().reset_index()
    tidy.columns = ["replication", "n", "mean_payoff"]
    tidy.to_csv(produces, index=False)
