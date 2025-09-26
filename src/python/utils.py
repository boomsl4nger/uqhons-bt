import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.optimize import check_grad

from BaseModel import BaseBradleyTerry

def make_win_matrix(df: DataFrame) -> DataFrame:
    teams = np.unique(df.iloc[:, 0:1].values.ravel())
    wins = DataFrame(0, index=teams, columns=teams, dtype=float)

    # TODO vectorise
    for _, row in df.iterrows():
        home, away, hscore, ascore = row
        
        if hscore > ascore:
            wins.loc[home, away] += 1
        elif ascore > hscore:
            wins.loc[away, home] += 1
        elif hscore == ascore: # Tie
            wins.loc[home, away] += 0.5
            wins.loc[away, home] += 0.5

    return wins

def make_time_blocked_win_matrix(df: DataFrame) -> dict[str: DataFrame]:
    pass

def make_venue_win_matrices(df: DataFrame) -> dict:
    teams = np.unique(df.iloc[:, 0:1].values.ravel())
    wins = {
        "home": DataFrame(0, index=teams, columns=teams, dtype=float),
        "away": DataFrame(0, index=teams, columns=teams, dtype=float),
        "neutral": DataFrame(0, index=teams, columns=teams, dtype=float)
    }

    # TODO vectorise
    for _, row in df.iterrows():
        home, away, hscore, ascore, is_neutral = row

        if hscore > ascore:
            winner, loser = home, away
            if is_neutral:
                wins['neutral'].loc[winner, loser] += 1
            else:
                wins['home'].loc[winner, loser] += 1
        elif ascore > hscore:
            winner, loser = away, home
            if is_neutral:
                wins['neutral'].loc[winner, loser] += 1
            else:
                wins['away'].loc[winner, loser] += 1
        else: # Tie
            if is_neutral:
                wins['neutral'].loc[home, away] += 0.5
                wins['neutral'].loc[away, home] += 0.5
            else:
                wins['home'].loc[home, away] += 0.5
                wins['away'].loc[away, home] += 0.5
            
    return wins

def make_time_blocked_venue_win_matrices(df: DataFrame) -> dict[str: dict[str: DataFrame]]:
    pass

def matrix_to_bin_counts(mat):
    pass

def make_victory_totals(data: DataFrame, home: bool = False) -> DataFrame:
    """Helper function to convert a win matrix into a table of total wins for each team. Also
    includes total losses, total games, and the win rate (%).

    TODO support for venue-based and time-based options

    Args:
        data (DataFrame): Win matrix. See `make_win_matrix()`.

    Returns:
        DataFrame: Table of victory totals per team.
    """
    wins = data.sum(axis=1)
    losses = data.sum(axis=0)
    total_games = wins + losses
    result = {
        "Team": data.index,
        "Wins": wins,
        "Losses": losses,
        "Total": total_games,
        "Win Rate": (wins / total_games).fillna(0)
    }

    return DataFrame(result).reset_index(drop=True)

def check_model_grad(model: BaseBradleyTerry) -> float:
    """Helper function to check the analytic score statistic against a numerical derivative. Wraps
    around the `scipy.check_grad` function.

    Args:
        model (BaseBradleyTerry): BT model to check.

    Returns:
        float: Difference (absolute error) in the gradients.
    """
    # x0 = np.zeros(model.n_params) + 1 / model.n_params
    rng = np.random.default_rng()
    x0 = rng.random(model.n_params) * 10
    return check_grad(model._log_likelihood, model._score, x0)



if __name__ in "__main__":
    # Some very basic test cases -> move to a test file
    path = "data/VFL1916.csv"
    vfl_df = pd.read_csv(path, header=0)
    vfl_mat = make_win_matrix(vfl_df)
    print(vfl_mat)
