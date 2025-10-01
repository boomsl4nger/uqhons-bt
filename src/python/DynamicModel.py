import numpy as np
from numpy import ndarray, exp, log
from pandas import DataFrame
from scipy.special import expit

from BaseModel import BaseDyBT
from StaticModel import *

# TODO handle teams coming in and out of data each year -> requirements for mats, etc
class DyVANBT(BaseDyBT):
    """Class for a discrete dyanmic 'vanilla' (VAN) Bradley-Terry model.

    The parameters are the team strengths `theta_{it}` on the log-scale for each team `i=1, ..., I`.
    In the discrete dynamic model, each team has a different strength parameter per time period, 
    such as years, `t=1,...,T`.

    Let's encode the params as
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T)]`

    TODO sort out matrix notation and correct vectorisation -> more for report
    TODO see `VANBT()` link for static version
    """
    def _set_data(self, data: dict[int: DataFrame]):
        # TODO data validation, check each matrix is right etc, check times make sense, better checking for names all same
        # TODO allow for non-pandas data matrix types, ie, generic iterables
        # Expecting: dict {year: win_matrix}
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a square DataFrame or dict of such.")
        
        self.data = data
        self.teams = list(data.values())[0].columns.tolist()
        self.n_teams = len(self.teams)
        self.times = sorted(list(data.keys()))
        self.n_times = len(self.times)
        self.n_params = self.get_n_params()

    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        for t, year in enumerate(self.times):
            cur_mat = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                        continue

                    wins = cur_mat.iloc[i, j]
                    theta_it = params[self._get_param_index(i, t)]
                    theta_jt = params[self._get_param_index(j, t)]

                    loglik += wins * log(self._calculate_prob(theta_it, theta_jt))

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        for t, year in enumerate(self.times):
            cur_mat = self.data[year]
            for i in range(self.n_teams):
                score_it = 0
                for j in range(self.n_teams):
                    if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                        continue

                    wins = cur_mat.iloc[i, j]
                    losses = cur_mat.iloc[j, i]
                    theta_it = params[self._get_param_index(i, t)]
                    theta_jt = params[self._get_param_index(j, t)]

                    score_it += wins - (wins + losses) * self._calculate_prob(theta_it, theta_jt)

                score[self._get_param_index(i, t)] = score_it

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def _get_param_index(self, i, t) -> int:
        return i + self.n_teams * t

    def get_param(self, team: str, time: str | int) -> float:
        """Get a model parameter. For the DyVANBT model, this is just the team strengths for each
        time block in the data.

        Args:
            team (str): Name of team. Raises an error if the team name is invalid.
            time (str | int): Name of time block. Raises an error if invalid.

        Returns:
            float: Parameter value.
        """
        self._check_fitted()
        
        try:
            team_idx = self.teams.index(team)
            time_idx = self.times.index(time)
        except ValueError:
            print(f"Team '{team}' or time '{time}' not found in the model.")
            raise

        return self.params[team_idx + self.n_teams * time_idx]

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times
    
    def summary(self) -> str:
        return super().summary()
    
    @staticmethod
    def _calculate_odds(it: float, jt: float) -> float:
        return VANBT._calculate_odds(it, jt)

    def get_odds(self, i: str, j: str, t: str | int) -> float:
        return self._calculate_odds(self.get_param(i, t), self.get_param(j, t))
    
    @staticmethod
    def _calculate_prob(it: float, jt: float) -> float:
        return expit(DyVANBT._calculate_odds(it, jt))

    def get_prob(self, i: str | int, j: str | int, t: str | int) -> float:
        return expit(self.get_odds(i, j, t))
