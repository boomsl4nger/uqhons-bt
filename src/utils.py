import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.differentiate import hessian
from scipy.linalg import norm
from scipy.optimize import check_grad
import seaborn as sns

from BaseModel import BaseBradleyTerry

# TODO func that converts venue matrices into win totals (ie generic VANBT support)
# will need for both static and discrete dyanmic mods
# TODO look at other util/support functions implemented by BT2 that would aid users

def make_win_matrix(df: DataFrame) -> DataFrame:
    """Create a win matrix for some given match data. The (i,j)-th element of the matrix is the
    number of times that team i beat team j.

    Note: currently expecting data with columns like `(teamA, teamB, scoreA, scoreB)`.

    TODO support non-pandas data input \\
    TODO better requirements, better validation, use first four columns explicitly

    Args:
        df (DataFrame): Match results data. See above format requirement.

    Returns:
        DataFrame: Win matrix.
    """
    teams = sorted(list(set(df.iloc[:, 0]).union(set(df.iloc[:, 1]))))
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
    """Creates win matrices for each time block (e.g. years) in some given match data. 
    
    See `make_win_matrix()`.

    Note: currently assuming data has columns like `(teamA, teamB, scoreA, scoreB, year)`.

    Returns:
        dict: dictionary of win matrices, where the keys are the unique time blocks (e.g. years).
    """
    all_teams = sorted(list(set(df.iloc[:, 0]).union(set(df.iloc[:, 1]))))
    times = df.iloc[:, -1].unique()
    matrices = {}

    for time in times:
        time_data = df[df.iloc[:, -1] == time].iloc[:, 0:4]
        m = make_win_matrix(time_data)
        
        # Reindex to ensure all teams are included
        m_full = m.reindex(index=all_teams, columns=all_teams, fill_value=0.0)
        
        matrices[time] = m_full

    return matrices

def make_venue_win_matrices(df: DataFrame) -> dict:
    """Create the venue win matrices and return in a dictionary. This function supports three types
    of venues: home, away, and neutral. The (i,j)-th element of each matrix represents the number of
    times team i beat team j at the respective venue type.

    Note: currently expecting data with columns `(teamA, teamB, scoreA, scoreB, is_neutral)`, where
    the last column is a Boolean indicating if the match should be interpreted as a neutral venue.
    Otherwise, assumes the first listed team is home by default.

    TODO is the bool column the best idea? or maybe a map (-1, 0, 1)?
    TODO allow last column to not be provided, and interpret as no neutral venues (still return mat)

    Args:
        df (DataFrame): Match results data. See above format requirements.

    Returns:
        dict: A dictionary of the win matrices, with keys `("home", "away", "neutral")`.
    """
    teams = sorted(list(set(df.iloc[:, 0]).union(set(df.iloc[:, 1]))))
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
    """Creates venue win matrices for each time block (e.g. years) in some given match data. 
    
    See `make_venue_win_matrix()`.

    Note: currently assuming data has columns like `(teamA, teamB, scoreA, scoreB, is_neutral, year)`.

    Returns:
        dict: dictionary of dictionary of venue win matrices, where the outer keys are the unique 
        time blocks (e.g. years) and inner keys are the venue types.
    """
    all_teams = sorted(list(set(df.iloc[:, 0]).union(set(df.iloc[:, 1]))))
    times = df.iloc[:, -1].unique()
    matrices = {}

    for time in times:
        time_data = df[df.iloc[:, -1] == time].iloc[:, 0:5]
        m = make_venue_win_matrices(time_data)
        
        # Reindex to ensure all teams are included
        for venue_type in m.keys():
            m[venue_type] = m[venue_type].reindex(index=all_teams, columns=all_teams, fill_value=0.0)
        
        matrices[time] = m

    return matrices

def matrix_to_bin_counts(mat: DataFrame) -> DataFrame:
    """Convert a win matrix to binomial counts. Contains the wins and losses for each unique pairing
    of teams in the data. The `wins` column denotes the wins of the first listed team `team1` 
    over the second `team2`, and similarly for `losses`.

    Args:
        mat (DataFrame): Win matrix. See `make_win_matrix()`.

    Returns:
        DataFrame: Table of binomial win and loss counts for each pairing of teams.
    """
    teams = mat.columns.to_list()
    n_teams = len(teams)
    result = []

    for i in range(n_teams):
        for j in range(i+1, n_teams):
            result.append([
                teams[i], teams[j], mat.iloc[i, j], mat.iloc[j, i]
            ])
    
    return DataFrame(result, columns=["team1", "team2", "wins", "losses"])

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

def check_model_hess(model: BaseBradleyTerry, show_warning: bool = False) -> tuple:
    """Helper function to check the analytic Hessian against a numerical calculation. Uses scipy's
    `differentiate.hessian` function for the latter. Difference is calculated by Frobenius norm.

    See: https://docs.scipy.org/doc//scipy/reference/generated/scipy.differentiate.hessian.html

    Args:
        model (BaseBradleyTerry): BT model to check. Must have called `_set_data()`.

    Returns:
        tuple: difference, H_analytic, H_numeric
    """
    rng = np.random.default_rng()
    x0 = rng.random(model.n_params) * 10
    hess_ana = model._hessian(x0)
    hess_num = hessian(model._log_likelihood, x0)

    if show_warning and not hess_num.success.all():
        print(f"Error in numerical Hessian calculation: status matrix below\n{hess_num.status}")

    hess_diff = norm(hess_ana - hess_num.ddf, ord="fro")
    # TODO look into np.allclose with tolerances
    # TODO better printing via rounding to sig figs

    return hess_diff, hess_ana, hess_num.ddf
