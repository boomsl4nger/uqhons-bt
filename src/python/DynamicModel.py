import numpy as np
from numpy import ndarray, exp, log
from pandas import DataFrame

from BaseModel import BaseDyBT

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
    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        get_param_index = lambda i, t: i + self.n_teams * t
        # TODO factor out get param index lambda, include functionality for get prob etc
        for t, year in enumerate(self.times):
            cur_mat = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                        continue

                    theta_it = params[get_param_index(i, t)]
                    theta_jt = params[get_param_index(j, t)]
                    wins = cur_mat.iloc[i, j]

                    loglik += wins * (theta_it - log(exp(theta_it) + exp(theta_jt)))

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        get_param_index = lambda i, t: i + self.n_teams * t
        for t, year in enumerate(self.times):
            cur_mat = self.data[year]
            for i in range(self.n_teams):
                score_it = 0
                for j in range(self.n_teams):
                    if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                        continue

                    theta_it = params[get_param_index(i, t)]
                    theta_jt = params[get_param_index(j, t)]
                    wins = cur_mat.iloc[i, j]
                    losses = cur_mat.iloc[j, i]

                    score_it += wins - (wins + losses) * (exp(theta_it) / (exp(theta_it) + exp(theta_jt)))

                score[get_param_index(i, t)] = score_it

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
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

    def get_n_params(self):
        return self.n_teams * self.n_times
    
    def summary(self):
        return super().summary()

    def get_odds(self, i: int, j: int, t: int):
        return self.params[i] - self.params[j]

    def get_prob(self, i: int, j: int, t: int):
        exp_i = exp(self.params[i])
        exp_j = exp(self.params[j])
        return exp_i / (exp_i + exp_j)
