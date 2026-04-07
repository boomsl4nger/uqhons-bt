from typing import Literal
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pandas import DataFrame, Series
from scipy.differentiate import hessian
from seaborn import FacetGrid
import seaborn as sns

from BaseModel import BaseBradleyTerry
from DynamicModel import *

RANDOM_SEED = 888


## BT model wrapper(s)
# TODO ranks option: ranks = strengths.rank(ascending=False)
# TODO refactor to be methods for each model for polymorphism
# TODO docstrings

def plot_strengths(
        model: BaseBradleyTerry, plot_type: Literal["dynamic", "average", "grid"] = "dynamic", **kwargs
    ) -> Axes:
    # Customise outside this fn: figsize, sns theme (ticks), despine, etc
    # https://seaborn.pydata.org/tutorial/aesthetics.html

    # COMMON HGA: add scale bar
    # TEAM-SPECIFIC HGA: multiple plots

    model._check_fitted()

    if isinstance(model, VANBT):
        _plot_strengths_van(model, plot_type, **kwargs)
    elif isinstance(model, CHABT):
        _plot_strengths_cha(model, plot_type, **kwargs)
    elif isinstance(model, TSABT):
        _plot_strengths_tsa(model, plot_type, **kwargs)
    elif isinstance(model, CHIBT):
        _plot_strengths_chi(model, plot_type, **kwargs)
    elif isinstance(model, TSIBT):
        _plot_strengths_tsi(model, plot_type, **kwargs)
    else:
        raise ValueError(f"Model {type(model).__name__} not recognised.")

def _plot_strengths_van(
        model: VANBT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        n_grid_cols: int = 4
    ) -> Axes:
    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"].sort_values()
        ax = _plot_strengths_static(rankings)
    else:
        rankings = model.get_ranking("Team", include_average=False)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, col_wrap=n_grid_cols)
        else:
            ax = _plot_strengths_dynamic(rankings)

    return ax

def _plot_strengths_cha(
        model: CHABT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        n_grid_cols: int = 4
    ) -> Axes:
    hga = model.get_hgas()["HGA"].values[0]
    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"].rename("Away").to_frame()
        rankings["Home"] = rankings["Away"] + hga
        rankings.sort_values("Away", inplace=True)
        ax = _plot_strengths_static(rankings)
    else:
        rankings = model.get_ranking("Team", include_average=False)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, hga, col_wrap=n_grid_cols)
        else:
            ax = _plot_strengths_dynamic(rankings, hga)

    return ax

def _plot_strengths_chi(
        model: CHIBT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        n_grid_cols: int = 4
    ) -> Axes:
    hga = model.get_hgas()["HGA"]
    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"].rename("Away").to_frame()
        for level in model.levels:
            rankings[f"Home_{level}"] = rankings["Away"] + hga[level]
        rankings.sort_values("Away", inplace=True)
        ax = _plot_strengths_static(rankings)
    else:
        rankings = model.get_ranking("Team", include_average=False)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, hga, col_wrap=n_grid_cols)
        else:
            ax = _plot_strengths_dynamic(rankings, hga)

    return ax

def _plot_strengths_tsa(
        model: TSABT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        n_grid_cols: int = 4, figsize: tuple = (14, 6), legend_loc: tuple = (-0.15, 1)
    ) -> Axes:
    hga = model.get_hgas()["HGA"]
    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"].rename("Away").to_frame()
        rankings["Home"] = rankings["Away"] + hga
        rankings.sort_values("Away", inplace=True)
        ax = _plot_strengths_static(rankings)
    else:
        rankings = model.get_ranking("Team", include_average=False)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, hga, col_wrap=n_grid_cols)
        else:
            # Plotting twice: baseline strengths then home-boosted strengths
            first_year = sorted(rankings.columns)[0]
            hue_order = rankings[first_year].sort_values(ascending=False).index.tolist()

            fig, ax = plt.subplots(ncols=2, figsize=figsize, sharey=True, sharex=True)

            # Baseline
            ax1 = _plot_strengths_dynamic(rankings, hue_order=hue_order, ax=ax[0])
            ax1.set_title("Away")

            handles, labels = ax1.get_legend_handles_labels()
            new_labels = [f"{team} ({hga.loc[team]:.2f})" for team in labels]
            ax1.legend(handles, new_labels, title="Team (HGA)")
            sns.move_legend(ax1, "upper right", bbox_to_anchor=legend_loc)

            # Home-boosted
            ax2 = _plot_strengths_dynamic(rankings.add(hga, axis=0), hue_order=hue_order, ax=ax[1])
            ax2.axhline(0, linestyle="--", alpha=0.3, c="k", zorder=1)
            ax2.set_title("Home")
            ax2.get_legend().remove()

    return ax

def _plot_strengths_tsi(
        model: TSIBT, plot_type: Literal["dynamic", "average", "grid"] = "dynamic",
        n_grid_cols: int = 4, figsize: tuple = (14, 6), legend_loc: tuple = (-0.15, 1)
    ) -> Axes:
    hga = model.get_hgas()
    if plot_type == "average":
        rankings = model.get_ranking("Team")["Average"].rename("Away").to_frame()
        for level in model.levels:
            rankings[f"Home_{level}"] = rankings["Away"] + hga[level]
        rankings.sort_values("Away", inplace=True)
        ax = _plot_strengths_static(rankings)
    else:
        rankings = model.get_ranking("Team", include_average=False)
        if plot_type == "grid":
            ax = _plot_strengths_dynamic_indiv(rankings, hga, col_wrap=n_grid_cols)
        else:
            # Plotting k times: baseline strengths then home-boosted strengths for each level
            first_year = sorted(rankings.columns)[0]
            hue_order = rankings[first_year].sort_values(ascending=False).index.tolist()

            fig, ax = plt.subplots(ncols=(model.n_levels+1), figsize=figsize, sharey=True, sharex=True)

            # Baseline
            ax1 = _plot_strengths_dynamic(rankings, hue_order=hue_order, ax=ax[0])
            ax1.set_title("Away")

            # handles, labels = ax1.get_legend_handles_labels()
            # new_labels = [f"{team} ({hga.loc[team]:.2f})" for team in labels]
            # ax1.legend(handles, new_labels, title="Team (HGA)")
            sns.move_legend(ax1, "upper right", bbox_to_anchor=legend_loc)

            # Home-boosted
            for i, level in enumerate(model.levels):
                ax2 = _plot_strengths_dynamic(rankings.add(hga[level], axis=0), hue_order=hue_order, ax=ax[i+1])
                ax2.axhline(0, linestyle="--", alpha=0.3, c="k", zorder=1)
                ax2.set_title(f"Home (level={level})")
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


def _plot_strengths_dynamic(ranking: DataFrame, hga: float | Series = None, hue_order: list = None, add_markers: bool = False, ax: Axes = None) -> Axes:
    # Legend ordering
    if hue_order is not None and not all(team in ranking.index for team in hue_order):
        raise ValueError(f"Teams in hue_order not found in ranking index!")
    if hue_order is None:
        first_year = sorted(ranking.columns)[0]
        hue_order = ranking[first_year].sort_values(ascending=False).index.tolist()

    # Plot team strengths
    rank_long = ranking.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    ax = sns.lineplot(
        data=rank_long, x="Year", y="Strength", hue="Team", hue_order=hue_order,
        marker=("o" if add_markers else None), linewidth=1, ax=ax
    )

    # Formatting stuff
    ax.set_xlabel("Time")
    ax.set_ylabel("Strength")
    plt.xticks(sorted(ranking.columns))

    # Add HGA scale legend via OffsetBox stacking
    if hga is not None:
        _add_hga_scalebar(ax, hga)

    return ax


def _plot_strengths_dynamic_indiv(ranking: DataFrame, errors: DataFrame = None, hga: float | Series = None, col_wrap: int = 4) -> FacetGrid:
    # Plot team strengths
    rank_long = ranking.reset_index(names="Team").melt(id_vars=["Team"], var_name="Year", value_name="Strength")
    g = sns.FacetGrid(rank_long, col="Team", col_wrap=col_wrap)
    g.map_dataframe(sns.lineplot, x="Year", y="Strength", marker="o", linewidth=1)

    g.set_axis_labels("Time", "Strength")
    # TODO x ticks are not integers

    return g


def _plot_strengths_static(ranking: Series | DataFrame, ax: Axes = None) -> Axes:
    if ax is None: ax = plt.gca()

    y = np.arange(len(ranking))
    if isinstance(ranking, Series): # VANBT: lollypop plot
        ax.hlines(y, xmin=0, xmax=ranking.values, color="k", alpha=0.3, lw=1, zorder=2)
        ax.scatter(ranking.values, y, zorder=3)
    else:                           # Other models: dumbbell plot
        ax.hlines(y, xmin=ranking.min(axis=1), xmax=ranking.max(axis=1), color="k", alpha=0.3, lw=2, zorder=2)
        palette = sns.color_palette(n_colors=len(ranking.columns))
        for col, c in zip(ranking.columns, palette):
            ax.scatter(ranking[col], y, label=col, color=c, zorder=3)
        ax.legend(loc="lower right")

    ax.set_yticks(y)
    ax.set_yticklabels(ranking.index)
    ax.axvline(0, linestyle="--", alpha=0.3, c="k", zorder=1)
    ax.set_xlabel("Strength")
    ax.set_ylabel("Team")

    return ax



## Other plotting fns
def make_competition_graph(data: DataFrame):
    # Set up the graph
    G = nx.DiGraph()
    teams = data.index.tolist()
    G.add_nodes_from(teams)

    for i in teams:
        for j in teams:
            w_ij = data.loc[i,j]
            if i != j and w_ij > 0:
                G.add_edge(i, j, weight=w_ij)

    return G

def plot_competition_graph(data: DataFrame):
    # TODO See: https://networkx.org/documentation/stable/index.html
    G = make_competition_graph(data)
    pos = nx.spring_layout(G, seed=RANDOM_SEED)  # force-directed layout
    weights = [G[u][v]["weight"] for u, v in G.edges]
    weights = [w / max(weights) * 5 for w in weights] # Normalise weights

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2000,
        font_size=10,
        arrows=True,
        width=weights
    )

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

def plot_win_prob_heatmap(model, team_of_interest: str, venue: str = "home") -> DataFrame:
    """Generates a heatmap of win probabilities for a team against all opponents over time."""
    opponents = [t for t in model.teams if t != team_of_interest]
    prob_matrix = np.zeros((len(opponents), model.n_times))
    
    for t_idx, time in enumerate(model.times):
        for o_idx, opponent in enumerate(opponents):
            prob_matrix[o_idx, t_idx] = model.get_prob(team_of_interest, opponent, time, venue)
            
    df_probs = DataFrame(prob_matrix, index=opponents, columns=model.times)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_probs, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5, vmin=0, vmax=1)
    plt.title(f"Win Probs for {team_of_interest} ({venue.capitalize()})")
    plt.xlabel("Time")
    plt.ylabel("Opponent")
    
    return df_probs

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
