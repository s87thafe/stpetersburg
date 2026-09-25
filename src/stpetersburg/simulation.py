"""Monte Carlo core for the St. Petersburg game.

Pure functions only: they take a generator and return arrays. No file I/O, no
global random state, no printing - that lives in the task modules. This is what
makes the simulation testable and the pytask graph honest about its inputs.
"""

from __future__ import annotations

import numpy as np

from stpetersburg.config import P_HEADS, PAYOFF_BASE, START_VALUE


def make_rng(seed: int) -> np.random.Generator:
    """Return the project's random generator.

    ``default_rng`` replaces the legacy ``np.random.*`` global state. Creating it
    once and passing it down means the entire project is a deterministic function
    of ``seed``, which ``np.random.geometric`` could never guarantee.
    """
    return np.random.default_rng(seed)


def draw_payoffs(
    n_rounds: int,
    rng: np.random.Generator,
    p: float = P_HEADS,
    base: float = PAYOFF_BASE,
) -> np.ndarray:
    """Draw ``n_rounds`` independent St. Petersburg payoffs.

    The number of flips until the first head is geometric: the running sum of
    Bernoulli draws can only reach 1 on the draw that *is* a 1, so "length until
    the sum is 1" is exactly the trial index of the first success. NumPy's
    ``geometric`` counts the successful trial itself, so its support starts at 1
    and the smallest possible payoff is ``base ** 1``.

    The cast to float64 is deliberate: an integer exponent would overflow int64
    silently past ``2 ** 62``, whereas float64 merely loses exact integer
    representation above ``2 ** 53``, which cannot shift a mean materially.
    """
    n_flips = rng.geometric(p, size=n_rounds)
    return base ** n_flips.astype(np.float64)


def running_mean(x: np.ndarray) -> np.ndarray:
    """Cumulative sample mean: element ``i`` is the mean of ``x[: i + 1]``.

    Lets one simulation of length N answer "what was the mean after n draws?"
    for every n <= N, instead of re-simulating once per sample size.
    """
    return np.cumsum(x) / np.arange(1, x.size + 1)


def convergence_grid(n_max: int, checkpoints: tuple[int, ...], n_points: int = 60) -> np.ndarray:
    """Log-spaced sample sizes, guaranteed to contain ``checkpoints``.

    Log spacing because the interesting behaviour spans five orders of magnitude;
    the union keeps the figure smooth while the paper table reads exact decades
    off the same simulation.
    """
    smooth = np.unique(np.geomspace(1, n_max, num=n_points).astype(np.int64))
    return np.unique(np.concatenate([smooth, np.asarray(checkpoints, dtype=np.int64)]))


def mc_ensemble_means(
    n_grid: np.ndarray,
    n_replications: int,
    rng: np.random.Generator,
    p: float = P_HEADS,
    base: float = PAYOFF_BASE,
) -> np.ndarray:
    """Sampling distribution of the mean payoff.

    Returns shape ``(n_replications, n_grid.size)``. Repeating the estimator is
    what makes this Monte Carlo rather than a single simulation: with St.
    Petersburg payoffs the *instability* of the sample mean is the finding, and
    one estimate per sample size cannot show it.

    Loops over replications instead of allocating one big array because
    ``n_replications * n_max`` floats would run to hundreds of megabytes; only
    the checkpoints are retained. Draws come from a single generator advanced
    sequentially, so replications are independent yet fully seed-determined.
    """
    n_grid = np.asarray(n_grid, dtype=np.int64)
    n_max = int(n_grid.max())
    out = np.empty((n_replications, n_grid.size))
    for r in range(n_replications):
        payoffs = draw_payoffs(n_max, rng, p=p, base=base)
        out[r] = running_mean(payoffs)[n_grid - 1]
    return out


def time_average_path(
    n_rounds: int,
    rng: np.random.Generator,
    cost: float,
    start_value: float = START_VALUE,
    p: float = P_HEADS,
    base: float = PAYOFF_BASE,
) -> tuple[np.ndarray, int]:
    """Wealth of one player over ``n_rounds`` sequential rounds.

    Wealth is carried forward by ``cumsum`` and the ticket ``cost`` is paid every
    round - this is what separates the time arm from the ensemble arm, where each
    player restarts from ``start_value``.

    Ruin is a modelling choice, not a technicality: once wealth reaches zero the
    player cannot buy the next ticket, so the path is frozen there. Without this
    the player borrows without limit and the time average is meaningless.

    Returns the wealth path and the round index of ruin (``-1`` if solvent).
    """
    payoffs = draw_payoffs(n_rounds, rng, p=p, base=base)
    wealth = start_value + np.cumsum(payoffs - cost)
    ruined = np.flatnonzero(wealth <= 0.0)
    t_ruin = int(ruined[0]) if ruined.size else -1
    if t_ruin >= 0:
        wealth[t_ruin:] = 0.0
    return wealth, t_ruin
