from typing import Literal
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from scipy.differentiate import hessian
from seaborn import FacetGrid
import seaborn as sns

from BaseModel import BaseBradleyTerry
from DynamicModel import *


## BT model wrapper(s)
# TODO ranks option: ranks = strengths.rank(ascending=False)
# TODO refactor to be methods for each model for polymorphism
# TODO docstrings

def plot_strengths(
        model: BaseBradleyTerry, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        with_errors: bool = False, n_grid_cols: int = 4
    ) -> Axes:
    # Customise outside this fn: figsize, sns theme (ticks), despine, etc
    # https://seaborn.pydata.org/tutorial/aesthetics.html

    # COMMON HGA: option to either add scale legend OR produce multiple plots
    # TEAM-SPECIFIC HGA: multiple plots

    model._check_fitted()

    if isinstance(model, VANBT):
        _plot_strengths_van(model, plot_type, with_errors, n_grid_cols)
    elif isinstance(model, CHABT):
        _plot_strengths_cha(model, plot_type, with_errors, n_grid_cols)
    elif isinstance(model, TSABT):
        _plot_strengths_tsa(model, plot_type, with_errors, n_grid_cols)
    else:
        raise ValueError(f"Model {type(model).__name__} not recognised.")

def _plot_strengths_van(
        model: VANBT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        with_errors: bool = False, n_grid_cols: int = 4
    ) -> Axes:

    # TODO
    errors = None
    # if with_errors:
    #     if model.errors is None: model._calculate_errors()
    #     n_strength_params = model.n_times * model.n_teams
    #     err_reshape = model.errors[:n_strength_params].reshape(model.n_times, model.n_teams).T
    #     errors = DataFrame(err_reshape, columns=model.times, index=model.teams)

    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"]
        ax = _plot_strengths_static(rankings, errors)
    else:
        rankings = model.get_ranking("Team", add_average=False)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, errors, col_wrap=n_grid_cols)
        else:
            ax = _plot_strengths_dynamic(rankings)

    return ax

def _plot_strengths_cha(
        model: CHABT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        with_errors: bool = False, n_grid_cols: int = 4
    ) -> Axes:
    errors = None
    hga = model.get_param("hga")

    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"]
        ax = _plot_strengths_static(rankings, errors, hga)
    else:
        rankings = model.get_ranking("Team").drop("Average", axis=1)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, errors, hga, col_wrap=n_grid_cols)
        else:
            ax = _plot_strengths_dynamic(rankings, hga)

    return ax

def _plot_strengths_tsa(
        model: TSABT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        with_errors: bool = False, n_grid_cols: int = 4
    ) -> Axes:
    errors = None
    hga = Series(model.params[-model.n_teams:], index=model.teams)

    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"]
        ax = _plot_strengths_static(rankings, errors, hga)
    else:
        rankings = model.get_ranking("Team").drop("Average", axis=1)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, errors, hga, col_wrap=n_grid_cols)
        else:
            # Plotting twice: baseline strengths then home-boosted strengths
            # TODO relplot would be cleaner
            fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)

            ax1 = ax[0] # Baseline plot
            _plot_strengths_dynamic(rankings, ax=ax1)
            ax1.set_title("Away")

            handles, labels = ax1.get_legend_handles_labels()
            new_labels = [f"{team} ({hga.loc[team]:.2f})" for team in labels]
            ax1.legend(handles, new_labels, title="Team (HGA)")
            sns.move_legend(ax1, "upper right", bbox_to_anchor=(-0.15, 1))

            ax2 = ax[1] # Home-boosted plot
            _plot_strengths_dynamic(rankings.add(hga, axis=0), ax=ax2)
            ax2.axhline(0, linestyle="--", alpha=0.3, c="k", zorder=1)
            ax2.set_title("Home")
            ax2.get_legend().remove()

    return ax


## Strength plotting fns

def _add_hga_scalebar(ax, hga: float | Series):
    # See: https://matplotlib.org/stable/api/offsetbox_api.html
    # TODO See: https://pypi.org/project/matplotlib-scalebar/

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredOffsetbox
    from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea
    from matplotlib.lines import Line2D

    if np.isscalar(hga):
        hga = Series({"Com.": hga})

    levels = list(hga.index)
    values = hga.values

    # Container for bars
    bars = []

    for lvl, val in zip(levels, values):

        # Box in data coordinates (so bar height is TRUE scale)
        bar_box = AuxTransformBox(ax.transData)

        # Base position (slightly above ymin to avoid clipping)
        ymin, ymax = ax.get_ylim()

        x0 = 0  # local coordinate inside AuxTransformBox

        # Vertical bar
        bar_box.add_artist(Line2D([x0, x0], [ymin, ymin + val],
                                  color="black", linewidth=2))

        # Caps
        cap_w = 0.02
        bar_box.add_artist(Line2D([x0 - cap_w, x0 + cap_w], [ymin, ymin],
                                  color="black", linewidth=2))
        bar_box.add_artist(Line2D([x0 - cap_w, x0 + cap_w], [ymin + val, ymin + val],
                                  color="black", linewidth=2))

        # Magnitude label on right
        mag = TextArea(f"{val:.2f}", textprops=dict(size=9, rotation=90, va="bottom", ha="center"))

        # Level label underneath
        lvl_label = TextArea(lvl, textprops=dict(size=9))
        
        stack_1 = HPacker(children=[bar_box, mag],          align="center", pad=0, sep=4)
        stack_2 = VPacker(children=[stack_1, lvl_label],    align="center", pad=0, sep=4)

        bars.append(stack_2)

    # Pack multiple bars side-by-side
    all_bars = HPacker(children=bars, align="center", pad=0, sep=15)

    # Add title above
    title = TextArea("HGA")

    legend_box = VPacker(children=[title, all_bars],
                         align="center", pad=0, sep=8)

    anchored_box = AnchoredOffsetbox(
        loc='lower left',
        child=legend_box,
        pad=0.5,
        borderpad=0.7,
        frameon=True,
        bbox_to_anchor=(1, 0),
        bbox_transform=ax.transAxes
    )

    ax.add_artist(anchored_box)


def _plot_strengths_dynamic(ranking: DataFrame, hga: float | Series = None, ax: Axes = None) -> Axes:
    # Plot team strengths
    rank_long = ranking.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    ax = sns.lineplot(
        data=rank_long, x="Year", y="Strength", hue="Team", 
        marker="o", linewidth=1, ax=ax
    )

    # Formatting stuff
    ax.set_xlabel("Time")
    ax.set_ylabel("Strength")
    plt.xticks(sorted(ranking.columns))
    sns.move_legend(ax, "upper right", bbox_to_anchor=(-0.15, 1))

    # Add HGA scale legend via OffsetBox stacking
    if hga is not None:
        _add_hga_scalebar(ax, hga)

    return ax


def _plot_strengths_dynamic_indiv(ranking: DataFrame, errors: DataFrame = None, hga: float | Series = None, col_wrap: int = 4) -> FacetGrid:
    # Plot team strengths
    rank_long = ranking.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    g = sns.FacetGrid(rank_long, col="Team", col_wrap=col_wrap)
    g.map_dataframe(sns.lineplot, x="Year", y="Strength", marker="o", linewidth=1)

    # Plot errors if provided
    # if errors is not None:
    #     err_long = errors.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    #     merged = rank_long.merge(err_long, on=["Team", "Year"])

    #     def _add_errorbars(data, **kws):
    #         plt.errorbar(
    #             data["Year"], data["Strength_x"], yerr=data["Strength_y"],
    #             fmt="none", alpha=0.4, zorder=-1)

    #     # TODO fix this
    #     g.map_dataframe(_add_errorbars, data=merged)

    g.set_axis_labels("Time", "Strength")
    # TODO x ticks are not integers

    return g


def _plot_strengths_static(ranking: Series, errors: Series = None, hga: float | Series = None) -> Axes:
    err_offset = -0.15

    # Massage data into one nice df
    df = ranking.rename("baseline").to_frame()

    if hga is None:         # Type of HGA
        df["home"] = np.nan
    elif np.isscalar(hga):
        df["home"] = df["baseline"] + hga
    else:
        df["home"] = df["baseline"] + hga.loc[df.index]
    if errors is not None:  # Errors are optional
        df["se"] = errors.loc[df.index]

    df = df.sort_values("baseline")
    
    # Plotting
    ax = plt.gca()
    y = np.arange(len(df))
    ax.scatter(df["baseline"], y, zorder=3, label="Neutral")

    if hga is not None:
        ax.scatter(df["home"], y, zorder=4, label="Home")
        for i, row in enumerate(df.itertuples()): # Dumbbell plot
            ax.plot([row.baseline, row.home], [i, i], alpha=0.3, zorder=2, c="k")
    else:
        for i, row in enumerate(df.itertuples()): # Lollypop plot
            ax.plot([0, row.baseline], [i, i], alpha=0.3, zorder=2, c="k")

    if errors is not None:
        ax.errorbar(
            df["baseline"], y + err_offset, xerr=df["se"],
            fmt="none", ecolor="0.6", elinewidth=1, capsize=2, zorder=1
        )


    ax.set_yticks(y)
    ax.set_yticklabels(df.index)

    ax.axvline(0, linestyle="--", alpha=0.3, c="k", zorder=1)
    ax.set_xlabel("Strength")
    ax.set_ylabel("Team")
    if hga is not None: ax.legend(loc="lower right")

    return ax



## Other plotting fns

def plot_competition(data):
    # TODO See: https://networkx.org/documentation/stable/index.html
    pass


def plot_hessian(mod: BaseBradleyTerry, diff: bool = False):
    """Plot analytic and numerical Hessians for comparison purposes. Model must be fitted already.

    Args:
        mod (BaseBradleyTerry): Model.
        diff (bool, optional): If true, plots the absolute difference matrix instead. Defaults to False.
    """
    mod._check_fitted()
    ana_hess = mod._hessian(mod.params)
    num_hess = hessian(mod._log_likelihood, mod.params).ddf

    if diff:
        plt.figure(figsize=(12, 8))
        sns.heatmap(np.abs(ana_hess - num_hess), annot=True, fmt=".1f")
        plt.title("Hessian diff")
    else:
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(ana_hess, annot=True, fmt=".1f")
        plt.title("Analytic")

        plt.subplot(1, 2, 2)
        sns.heatmap(num_hess, annot=True, fmt=".1f")
        plt.title("Numeric")
