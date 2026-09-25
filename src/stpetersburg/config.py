"""Project-wide paths and simulation parameters.

Every number that defines the experiment lives here, so a run is described by
this file plus the code hash - not by whatever was typed into a notebook cell.
"""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "output" / "figures"

#: Master seed. Every draw in the project descends from this one integer.
SEED = 25_092_026

# --- Game definition -------------------------------------------------------
P_HEADS = 0.5  # probability of the coin landing heads on any flip
PAYOFF_BASE = 2.0  # payoff is PAYOFF_BASE ** (number of flips)
START_VALUE = 100.0  # initial wealth of a player
COST_CONSTANT = 2.0  # price paid to play one round

# --- Convergence experiment ------------------------------------------------
N_REPLICATIONS = 200  # independent repetitions of the whole experiment
N_MAX = 100_000  # largest sample size on the convergence grid
N_CHECKPOINTS = (10, 100, 1_000, 10_000, 100_000)  # the notebook's draws_vec
