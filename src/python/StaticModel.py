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


class CHABT(BaseBradleyTerry):
    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                home_wins = self.data["home"].iloc[i, j]
                away_wins = self.data["away"].iloc[i, j]
                neutral_wins = self.data["neutral"].iloc[i, j]

                loglik += home_wins * (params[i] + params[-1] - log(exp(params[i] + params[-1]) + exp(params[j]))) \
                    + away_wins * (params[i] - log(exp(params[i]) + exp(params[j] + params[-1]))) \
                    + neutral_wins * (params[i] - log(exp(params[i]) + exp(params[j])))

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        for i in range(self.n_teams):
            score_i = 0
            for j in range(self.n_teams):
                if i == j:
                    continue

                home = self.data["home"]
                away = self.data["away"]
                neutral = self.data["neutral"]

                score_i += home.iloc[i, j] + away.iloc[i, j] + neutral.iloc[i, j] \
                    - (home.iloc[i, j] + away.iloc[j, i]) * (exp(params[i] + params[-1]) / (exp(params[i] + params[-1]) + exp(params[j]))) \
                    - (home.iloc[j, i] + away.iloc[i, j]) * (exp(params[i]) / (exp(params[i]) + exp(params[j] + params[-1]))) \
                    - (neutral.iloc[i, j] + neutral.iloc[j, i]) * (exp(params[i]) / (exp(params[i]) + exp(params[j])))

            score[i] = score_i

        for i in range(self.n_teams):
            for j in range(self.n_teams):
                score[-1] += self.data["home"].iloc[i, j] * (exp(params[j]) / (exp(params[i] + params[-1]) + exp(params[j]))) \
                    - self.data["away"].iloc[i, j] * (exp(params[j] + params[-1]) / (exp(params[i]) + exp(params[j] + params[-1])))

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def _set_data(self, data: dict[str: DataFrame]):
        # TODO data validation
        # TODO allow for non-pandas data matrix types, ie, generic iterables
        # Assume dict of DFs for now
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a dictionary of square DataFrames.")
        
        assert list(data.keys()) == ["home", "away", "neutral"]
        assert all([i.shape[0] == i.shape[1] for i in data.values()])
        
        self.data = data
        self.teams = data["home"].columns.tolist()
        self.n_teams = len(self.teams)
        self.n_params = self.get_n_params()
    
    def get_n_params(self):
        return self.n_teams + 1

    def get_ranking(self):
        assert self.params is not None

        results = {
            "Team": self.teams,
            "Ability": self.params[0:self.n_teams]
        }

        return DataFrame(results).sort_values(by="Ability", ascending=False)
    
    def get_odds(self, i, j, venue = "home"):
        return self.params[i] - self.params[j]

    def get_prob(self, i, j, venue = "home"):
        exp_i = exp(self.params[i])
        exp_j = exp(self.params[j])
        return exp_i / (exp_i + exp_j)


class CHIBT(BaseBradleyTerry):
    def get_n_params(self):
        return self.n_teams + 1

    def _init_params(self):
        return np.zeros(self.n_params)

    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_odds(self, i, j):
        pass

    def get_prob(self, i, j):
        pass


class TSHBT(BaseBradleyTerry):
    def get_n_params(self):
        return self.n_teams + 1

    def _init_params(self):
        return np.zeros(self.n_params)

    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_odds(self, i, j):
        pass

    def get_prob(self, i, j):
        pass


class HIEBT(BaseBradleyTerry):
    def get_n_params(self):
        return self.n_teams + 1

    def _init_params(self):
        return np.zeros(self.n_params)

    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_odds(self, i, j):
        pass

    def get_prob(self, i, j):
        pass
