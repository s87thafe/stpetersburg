"""Figure builders for the paper.

Translated from the notebook's ``plot_ensemble_means``: a quantile fan around the
median of the replications. Design is unchanged - single hue, two nested bands,
median on top - only the pieces a notebook does not need were added: explicit
figure dimensions (a ``.show()`` in Jupyter sizes itself, a PDF cannot) and no
``.show()`` call, since the task writes the file.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

#: Points, not pixels: kaleido maps 1 px to 1 pt in PDF, so these are the
#: figure's real dimensions on the page (~6.4in x 4.0in).
WIDTH = 460
HEIGHT = 290


def _band(n_grid: np.ndarray, lower: np.ndarray, upper: np.ndarray, fillcolor: str, name: str):
    """One filled quantile band.

    Drawn as a single closed polygon: the upper edge forward, the lower edge
    reversed, so ``fill="toself"`` closes it without a baseline.
    """
    return go.Scatter(
        x=np.concatenate([n_grid, n_grid[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor=fillcolor,
        line={"width": 0},
        hoverinfo="skip",
        name=name,
    )


def plot_ensemble_means(
    n_grid: np.ndarray,
    means: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.75, 0.95),
) -> go.Figure:
    """Spread of the sample mean payoff across replications, against sample size.

    ``means`` has shape ``(n_replications, n_grid.size)``. Quantiles are taken
    down the replication axis, so each vertical slice is the sampling
    distribution of the mean at that sample size.
    """
    n_grid = np.asarray(n_grid)
    q = np.quantile(means, quantiles, axis=0)
    median = np.median(means, axis=0)

    fig = go.Figure()
    fig.add_trace(_band(n_grid, q[0], q[3], "rgba(70,110,170,0.15)", "5-95%"))
    fig.add_trace(_band(n_grid, q[1], q[2], "rgba(70,110,170,0.30)", "25-75%"))
    fig.add_trace(
        go.Scatter(
            x=n_grid,
            y=median,
            mode="lines",
            line={"color": "rgb(30,60,120)", "width": 2},
            name="median",
        )
    )

    fig.update_layout(
        width=WIDTH,
        height=HEIGHT,
        xaxis={"type": "log", "title": "sample size N"},
        yaxis={"title": "sample mean payoff"},
        template="simple_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        margin={"l": 60, "r": 20, "t": 20, "b": 50},
    )
    return fig
