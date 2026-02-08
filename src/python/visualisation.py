from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from scipy.differentiate import hessian
from seaborn import FacetGrid
import seaborn as sns

from BaseModel import BaseBradleyTerry


## BT model wrapper(s)
# TODO ranks option: ranks = strengths.rank(ascending=False)

def plot_strengths(
        model: BaseBradleyTerry, with_errors: bool = False, hga: float | Series = None,
        as_average: bool = False, as_facet: bool = False, n_facet_cols: int = 4
    ) -> Axes:
    # Customise outside this fn: figsize, sns theme (ticks), despine, etc
    # https://seaborn.pydata.org/tutorial/aesthetics.html

    # COMMON HGA: option to either add scale legend OR produce multiple plots
    # TEAM-SPECIFIC HGA: multiple plots

    model._check_fitted()

    errors = None
    if with_errors:
        if model.errors is None: model._calculate_errors()
        n_strength_params = model.n_times * model.n_teams
        err_reshape = model.errors[:n_strength_params].reshape(model.n_times, model.n_teams).T
        errors = DataFrame(err_reshape, columns=model.times, index=model.teams)

    if as_average:
        rankings = model.get_ranking("Team")["Average"]
        ax = _plot_strengths_static(rankings, errors, hga)
    else:
        rankings = model.get_ranking("Team").drop("Average", axis=1)
        if as_facet:
            ax = _plot_strengths_dynamic_indiv(rankings, errors, col_wrap=n_facet_cols)
        else:
            ax = _plot_strengths_dynamic(rankings, errors)

    return ax


## Strength plotting fns

# TODO customisable features? palette, swarming (when lots of teams?)
def _plot_strengths_dynamic(ranking: DataFrame, errors: DataFrame = None) -> Axes:
    # Plot team strengths
    rank_long = ranking.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    ax = sns.lineplot(
        data=rank_long, x="Year", y="Strength", hue="Team", 
        marker="o", linewidth=1)
    
    # Plot errors if provided
    if errors is not None:
        err_long = errors.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
        merged = rank_long.merge(err_long, on=["Team", "Year"])
        for _, g in merged.groupby("Team"):
            ax.errorbar(
                g["Year"], g["Strength_x"], yerr=g["Strength_y"],
                fmt="none", alpha=0.4, zorder=-1)

    # Formatting stuff
    ax.set_xlabel("Time")
    ax.set_ylabel("Strength")
    plt.xticks(sorted(ranking.columns))
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(-0.25, 1))

    return ax

def _plot_strengths_dynamic_indiv(ranking: DataFrame, errors: DataFrame = None, col_wrap: int = 4) -> FacetGrid:
    # Plot team strengths
    rank_long = ranking.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    g = sns.FacetGrid(rank_long, col="Team", col_wrap=col_wrap)
    g.map_dataframe(sns.lineplot, x="Year", y="Strength", marker="o", linewidth=1)

    # Plot errors if provided
    if errors is not None:
        err_long = errors.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
        merged = rank_long.merge(err_long, on=["Team", "Year"])

        def _add_errorbars(data, **kws):
            plt.errorbar(
                data["Year"], data["Strength_x"], yerr=data["Strength_y"],
                fmt="none", alpha=0.4, zorder=-1)

        # TODO fix this
        g.map_dataframe(_add_errorbars, data=merged)

    g.set_axis_labels("Time", "Strength")
    # TODO x ticks not integers

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
        for i, row in enumerate(df.itertuples()):
            ax.plot(
                [row.baseline, row.home], [i, i],
                alpha=0.4, zorder=2, c="k"
            )

    if errors is not None:
        ax.errorbar(
            df["baseline"], y + err_offset, xerr=df["se"],
            fmt="none", ecolor="0.6", elinewidth=1, capsize=2, zorder=1
        )


    ax.set_yticks(y)
    ax.set_yticklabels(df.index)

    # ax.axvline(0, linestyle="--", alpha=0.4)
    ax.set_xlabel("Strength")
    ax.set_ylabel("Team")
    if hga is not None: ax.legend()

    return ax


## Other plotting fns

def plot_competition(data):
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
