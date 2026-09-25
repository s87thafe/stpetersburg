# St. Petersburg paradox and ergodicity

Monte Carlo study of how the sample mean payoff of the St. Petersburg game
behaves as the number of rounds grows.

## Reproducing

```bash
uv sync          # exact environment from uv.lock
uv run pytask    # run the simulation
```

Deterministic: every draw descends from the single `SEED` in
`src/stpetersburg/config.py`, so `uv run pytask --force` reproduces a
byte-identical CSV.

## Layout

| Path | Role |
|---|---|
| `src/stpetersburg/config.py` | paths and experiment parameters, incl. the seed |
| `src/stpetersburg/simulation.py` | pure simulation functions (no I/O, no global RNG) |
| `src/stpetersburg/plotting.py` | figure builders returning `plotly` figures |
| `src/stpetersburg/task_*.py` | pytask stages wiring inputs to outputs |
| `data/processed/` | generated CSVs — never edit by hand |
| `output/figures/` | generated PDFs — never edit by hand |
| `paper/` | the LaTeX source |
| `notebooks/` | exploratory scratch, not part of the build |

## Notes

Task modules declare `simulation.py`, `plotting.py` and `config.py` as explicit
dependencies.
pytask hashes a task module's own source and its declared files, but does not
follow first-party imports — without those arguments, changing the seed would
leave stale outputs in place and pytask would report "skipped because unchanged".

`kaleido` is pinned to `==0.2.1` and `plotly` to `<7`. Kaleido
1.x drops the bundled Chromium and shells out to a system Chrome install, which
no lockfile can pin; the 0.2.1 wheel carries its own browser.
