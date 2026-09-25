# St. Petersburg paradox and ergodicity

Monte Carlo comparison of the **ensemble average** (many players, one round each)
against the **time average** (one player, many sequential rounds) for the
St. Petersburg game.

## Reproducing everything

```bash
uv sync          # exact environment from uv.lock
uv run pytask    # simulate -> figures -> tables
latexmk          # build the paper
```

`uv run pytask` is the whole pipeline. It is deterministic: every draw descends
from the single `SEED` in `src/stpetersburg/config.py`, so a forced re-run
(`uv run pytask --force`) reproduces byte-identical CSVs and tables.

## Layout

| Path | Role |
|---|---|
| `src/stpetersburg/config.py` | all paths and experiment parameters, incl. the seed |
| `src/stpetersburg/simulation.py` | pure simulation functions (no I/O, no global RNG) |
| `src/stpetersburg/plotting.py` | figure builders returning `plotly` figures |
| `src/stpetersburg/task_*.py` | pytask stages wiring inputs to outputs |
| `data/processed/` | generated CSVs — never edit by hand |
| `output/figures/`, `output/tables/` | generated PDFs and LaTeX fragments |
| `paper/` | the LaTeX source |
| `notebooks/` | exploratory scratch, not part of the build |

Anything under `data/processed/`, `output/figures/` or `output/tables/` is a
build product. Edit the task that generates it, not the file.

## Notes on the pinned dependencies

`kaleido` is pinned to `==0.2.1` and `plotly` to `<7`. Kaleido 1.x drops the
bundled Chromium and shells out to a system Chrome install, which no lockfile can
pin — `uv sync` would resolve cleanly and then fail at figure export on a fresh
machine. The 0.2.1 wheel carries its own browser, so the lockfile is sufficient.
It emits a deprecation warning on every export; that is expected.

Task modules declare `simulation.py`, `plotting.py` and `config.py` as explicit
dependencies. pytask hashes a task module's own source and its declared files,
but does not follow first-party imports — without those lines, changing the seed
or a figure's colours would leave stale outputs in place.
