"""Figure builders for the paper.

Target is a static PDF inside a LaTeX document, so two passes from the general
data-viz method are deliberately skipped: there is no hover layer (a printed
figure has no pointer) and no dark mode (a PDF renders one way). The form
choice, palette, mark specs and direct-labelling rules all still apply.

Colours are categorical slots 1-3 of the reference palette, used unchanged.
That triple is the documented all-pairs-validated subset, so it is safe for
every chart form; slot 3 (aqua) sits below 3:1 contrast on a light surface,
which is why every series carries a visible direct label rather than relying on
colour alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Reference palette, light surface ---------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
DEEMPHASIS = "#c3c2b7"

FONT = "Helvetica, Arial, sans-serif"
#: Points, not pixels: kaleido maps 1 px to 1 pt in PDF, so these are the
#: figure's real dimensions on the page (~6.4in x 4.0in).
WIDTH = 460
HEIGHT = 290


def _base_layout(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    """Shared chrome: recessive grid and axes, text in ink tokens not series colour."""
    fig.update_layout(
        width=WIDTH,
        height=HEIGHT,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "size": 9, "color": TEXT_SECONDARY},
        margin={"l": 52, "r": 64, "t": 28, "b": 42},
        showlegend=False,
    )
    fig.update_xaxes(
        title={"text": x_title, "font": {"size": 9, "color": TEXT_SECONDARY}},
        gridcolor=GRIDLINE,
        griddash="dot",
        zeroline=False,
        linecolor=BASELINE,
        tickcolor=BASELINE,
        tickfont={"size": 8, "color": MUTED},
    )
    fig.update_yaxes(
        title={"text": y_title, "font": {"size": 9, "color": TEXT_SECONDARY}},
        gridcolor=GRIDLINE,
        griddash="dot",
        zeroline=False,
        linecolor=BASELINE,
        tickcolor=BASELINE,
        tickfont={"size": 8, "color": MUTED},
    )
    return fig


def _label(
    fig: go.Figure, x, y, text: str, color: str, log_x: bool = False, log_y: bool = False
) -> None:
    """Direct label pinned just right of a line's final point.

    Annotation coordinates on a log axis are given in log10 units, not data
    units - passing a raw value there places the label at 10**value, i.e. off
    the canvas entirely. The flags make that conversion explicit at each call
    site rather than leaving it to be remembered.
    """
    fig.add_annotation(
        x=np.log10(x) if log_x else x,
        y=np.log10(y) if log_y else y,
        text=text,
        showarrow=False,
        xanchor="left",
        xshift=6,
        font={"family": FONT, "size": 8, "color": color},
    )


def ensemble_convergence(means: pd.DataFrame, n_paths_shown: int) -> go.Figure:
    """Figure 1: the sample mean does not settle.

    Emphasis form, not categorical: the median across replications is the point,
    the individual replications are context. Drawing them in the de-emphasis grey
    shows the jumps - each one is a single enormous payoff entering the average -
    without letting eight competing hues bury the summary line.

    Log-log because both axes span five orders of magnitude.
    """
    wide = means.pivot(index="replication", columns="n", values="mean_payoff")
    n_values = wide.columns.to_numpy()
    fig = go.Figure()

    for i, (_, path) in enumerate(wide.head(n_paths_shown).iterrows()):
        fig.add_trace(
            go.Scatter(
                x=n_values,
                y=path.to_numpy(),
                mode="lines",
                line={"color": DEEMPHASIS, "width": 1},
                hoverinfo="skip",
                showlegend=False,
                name="single replication" if i == 0 else None,
            )
        )

    median = wide.median(axis=0).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=n_values,
            y=median,
            mode="lines",
            line={"color": SERIES[0], "width": 2},
            hoverinfo="skip",
        )
    )

    _label(fig, n_values[-1], median[-1], "median", SERIES[0], log_x=True, log_y=True)
    _label(
        fig,
        n_values[-1],
        wide.head(n_paths_shown).iloc[:, -1].max(),
        f"{n_paths_shown} single<br>replications",
        MUTED,
        log_x=True,
        log_y=True,
    )

    _base_layout(fig, "number of rounds played (n)", "sample mean payoff")
    # dtick=1 on a log axis labels decades only. Plotly's default adds mantissa
    # ticks ("2", "5"), which read as data values sitting beside the series.
    fig.update_xaxes(type="log", dtick=1)
    fig.update_yaxes(type="log", dtick=1)
    return fig


def time_paths(paths: pd.DataFrame) -> go.Figure:
    """Figure 2: one player through time, at three ticket prices.

    Categorical here - the three cost levels *are* the subject - and all three
    series share one axis (wealth), so no second y-scale is needed or allowed.
    Log y because a solvent path grows by four orders of magnitude.

    A ruined path is clipped at the last solvent round: zero cannot be drawn on a
    log axis, and extending a flat line at zero would imply the player keeps
    playing after bankruptcy.
    """
    fig = go.Figure()
    for i, (cost, group) in enumerate(paths.groupby("cost", sort=True)):
        colour = SERIES[i % len(SERIES)]
        solvent = group.loc[group["wealth"] > 0]
        fig.add_trace(
            go.Scatter(
                x=solvent["round"].to_numpy(),
                y=solvent["wealth"].to_numpy(),
                mode="lines",
                line={"color": colour, "width": 2},
                hoverinfo="skip",
            )
        )
        if not solvent.empty:
            last = solvent.iloc[-1]
            ruined = len(solvent) < len(group)
            if ruined:
                # A player ruined after a handful of rounds draws a line too
                # short to see. The marker is the only thing that makes the
                # fastest-failing series legible at all; the surface-coloured
                # ring keeps it readable where it overlaps the axis.
                fig.add_trace(
                    go.Scatter(
                        x=[last["round"]],
                        y=[last["wealth"]],
                        mode="markers",
                        marker={"color": colour, "size": 8, "line": {"color": SURFACE, "width": 2}},
                        hoverinfo="skip",
                    )
                )
            suffix = " (ruined)" if ruined else ""
            _label(fig, last["round"], last["wealth"], f"cost {cost:g}{suffix}", colour, log_y=True)

    _base_layout(fig, "round", "wealth")
    fig.update_yaxes(type="log", dtick=1)
    return fig
