import numpy as np
from numpy import exp, log

from BaseModel import BaseBradleyTerry

class VanillaBT(BaseBradleyTerry):
    def __init__(self, data):
        super().__init__(data)
        self.nparams = self.teams.size
        self.params = self._init_params()

    def _init_params(self):
        return np.zeros(self.nparams)

    def _log_likelihood(self, params: list) -> float:
        loglik = 0
        for i in range(self.nteams):
            for j in range(self.nteams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                wins = self.data.iloc[i, j]
                loglik += wins * (params[i] - log(exp(params[i]) + exp(params[j])))

        return -loglik

    def _score(self, params: list) -> np.ndarray:
        score = np.zeros(self.nparams)
        for i in range(self.nteams):
            score_i = 0
            for j in range(self.nteams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                wins = self.data.iloc[i, j]
                losses = self.data.iloc[j, i]
                score_i += (wins + losses) * (exp(params[i]) / (exp(params[i]) + exp(params[j])))

            score[i] = np.sum(self.data.iloc[i, :]) - score_i

        return -score

    def _hessian(self, params: list) -> list:
        raise NotImplementedError()