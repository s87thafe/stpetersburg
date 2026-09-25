"""Simulation stage: draw everything once, write tidy CSVs.

pytask discovers ``task_*`` functions in ``task_*.py`` modules. Each output is
annotated ``Annotated[Path, Product]``, which is how pytask learns the dependency
graph: it hashes the products and the task's own source, so editing
``simulation.py`` invalidates these CSVs and re-runs the figures downstream.
Simulating here and plotting elsewhere means a figure tweak never re-draws random
numbers - the expensive, seed-sensitive step stays cached.

``simulation.py`` and ``config.py`` are declared as dependencies because pytask
does not follow first-party imports: changing the seed or the payoff rule must
invalidate these CSVs, and nothing else would tell it to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from stpetersburg.config import (
    COSTS,
    DATA_PROCESSED,
    N_CHECKPOINTS,
    N_MAX,
    N_REPLICATIONS,
    N_ROUNDS,
    SEED,
    SRC,
)
from stpetersburg.simulation import (
    convergence_grid,
    make_rng,
    mc_ensemble_means,
    time_average_path,
)


def task_ensemble_means(
    simulation_module: Path = SRC / "simulation.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = DATA_PROCESSED / "ensemble_means.csv",
) -> None:
    """Ensemble arm: N independent players, one round each, repeated."""
    rng = make_rng(SEED)
    n_grid = convergence_grid(N_MAX, N_CHECKPOINTS)
    means = mc_ensemble_means(n_grid, N_REPLICATIONS, rng)
    # Tidy, not wide: one row per (replication, n) survives adding parameters
    # later and is what the plotting layer expects.
    tidy = pd.DataFrame(means, columns=n_grid).stack().reset_index()
    tidy.columns = ["replication", "n", "mean_payoff"]
    tidy.to_csv(produces, index=False)


def task_time_paths(
    simulation_module: Path = SRC / "simulation.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = DATA_PROCESSED / "time_paths.csv",
) -> None:
    """Time arm: one representative wealth path per ticket price.

    Thinned to every 10th round beyond round 100. A PDF cannot resolve 10,000
    points across 6 inches, and the untouched file would be 30x larger for no
    visible gain - but uniform thinning would erase the early rounds, where an
    expensive ticket bankrupts the player in under ten rounds. Keeping the first
    100 rounds intact preserves that.
    """
    rng = make_rng(SEED)
    frames = []
    for cost in COSTS:
        wealth, _ = time_average_path(N_ROUNDS, rng, cost=cost)
        rounds = np.arange(1, wealth.size + 1)
        keep = np.union1d(np.arange(min(100, wealth.size)), np.arange(0, wealth.size, 10))
        frames.append(
            pd.DataFrame(
                {
                    "cost": cost,
                    "round": rounds[keep],
                    "wealth": wealth[keep],
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(produces, index=False)


def task_time_summary(
    simulation_module: Path = SRC / "simulation.py",
    config_module: Path = SRC / "config.py",
    produces: Annotated[Path, Product] = DATA_PROCESSED / "time_summary.csv",
) -> None:
    """Ruin frequency and typical final wealth per ticket price.

    A separate generator seeded from SEED, so this task's draws are reproducible
    on its own and do not depend on whether the other tasks ran first.
    """
    rng = make_rng(SEED)
    records = []
    for cost in COSTS:
        finals = np.empty(N_REPLICATIONS)
        ruins = 0
        for r in range(N_REPLICATIONS):
            wealth, t_ruin = time_average_path(N_ROUNDS, rng, cost=cost)
            finals[r] = wealth[-1]
            ruins += t_ruin >= 0
        records.append(
            {
                "cost": cost,
                "median_final_wealth": float(np.median(finals)),
                "ruin_fraction": ruins / N_REPLICATIONS,
                "median_gain_per_round": float(np.median(finals) - 100.0) / N_ROUNDS,
            }
        )
    pd.DataFrame(records).to_csv(produces, index=False)
