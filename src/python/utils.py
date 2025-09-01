import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import check_grad

from BaseModel import BaseBradleyTerry

def data_to_win_matrix(df: DataFrame) -> DataFrame:
    teams = np.unique(df.iloc[:, 0:1].values.ravel())
    wins = DataFrame(0, index=teams, columns=teams, dtype=float)

    # TODO vectorise
    # TODO handle need for blocking by home wins/losses
    for _, row in df.iterrows():
        home, away, hscore, ascore = row
        
        if hscore > ascore:
            wins.loc[home, away] += 1
        elif ascore > hscore:
            wins.loc[away, home] += 1
        elif hscore == ascore:
            wins.loc[home, away] += 0.5
            wins.loc[away, home] += 0.5

    return wins

def matrix_to_counts(mat):
    pass

def make_victory_totals(data: DataFrame, home: bool = False) -> DataFrame:
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
    x0 = np.zeros(model.nparams) + 1 / model.nparams
    error = check_grad(model._log_likelihood, model._score, x0)
    return error



if __name__ in "__main__":
    # Some very basic test cases -> move to a test file
    path = "data/VFL1916.csv"
    vfl_df = pd.read_csv(path, header=0)
    vfl_mat = data_to_win_matrix(vfl_df)
    print(vfl_mat)
