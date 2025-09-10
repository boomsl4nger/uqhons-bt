import numpy as np
from numpy import ndarray, exp, log
from pandas import DataFrame

from BaseModel import BaseBradleyTerry

class VANBT(BaseBradleyTerry):
    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                wins = self.data.iloc[i, j]
                loglik += wins * (params[i] - log(exp(params[i]) + exp(params[j])))

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        for i in range(self.n_teams):
            score_i = 0
            for j in range(self.n_teams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                wins = self.data.iloc[i, j]
                losses = self.data.iloc[j, i]
                score_i += wins - (wins + losses) * (exp(params[i]) / (exp(params[i]) + exp(params[j])))

            score[i] = score_i

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def _set_data(self, data: DataFrame):
        # TODO data validation
        # TODO allow for non-pandas data matrix types, ie, generic iterables
        # Assume data is a pandas df for now
        if not isinstance(data, DataFrame) and data.shape[0] != data.shape[1]:
            raise ValueError("Input 'data' must be a square DataFrame of dict of such.")
        
        self.data = data
        self.teams = data.columns.tolist()
        self.n_teams = len(data.columns)
        self.n_params = self.get_n_params()

    def get_n_params(self):
        return self.n_teams

    def get_ranking(self):
        assert self.params is not None

        results = {
            "Team": self.teams,
            "Ability": self.params
        }

        return DataFrame(results).sort_values(by="Ability", ascending=False)
    
    def get_odds(self, i, j):
        return self.params[i] - self.params[j]

    def get_prob(self, i, j):
        exp_i = exp(self.params[i])
        exp_j = exp(self.params[j])
        return exp_i / (exp_i + exp_j)
        raise NotImplementedError()