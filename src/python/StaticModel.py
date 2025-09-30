import numpy as np
from numpy import ndarray, exp, log
from pandas import DataFrame
from scipy.special import expit

from BaseModel import BaseBradleyTerry

# TODO hessians and summaries (init errors on summary call)
# TODO type hint fields? eg self.data: dict, self.teams: list
# TODO enum for things like venue types
class VANBT(BaseBradleyTerry):
    """Class for a 'vanilla' (VAN) Bradley-Terry model. This is the original form of the model as proposed
    by Bradley & Terry (1952). The term 'vanilla' follows from Jones (2021, unpublished) to describe
    this simplest BT model formulation.

    The parameters are the team strengths `theta_i` on the log-scale for each team `i=1, ..., I`.

    TODO define a nice and small outline for these class descriptions.
    TODO docstrings for n_params/get_param/odds/prob methods, since these are model-specific
    """
    def _set_data(self, data: DataFrame):
        """Set class variables based on the given data. TODO specify data requirements

        Args:
            data (DataFrame): Data for the VANBT model.
        """
        # TODO data validation
        # TODO allow for non-pandas data matrix types, ie, generic iterables?
        # Assume data is a pandas df for now
        assert isinstance(data, DataFrame)
        assert data.shape[0] == data.shape[1]
        
        self.data = data
        self.teams = data.columns.tolist()
        self.n_teams = len(data.columns)
        self.n_params = self.get_n_params()

    def _log_likelihood(self, params: ndarray) -> float:
        # TODO think about using team names w/ .loc? is this handled enough in _set_data?
        loglik = 0
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                wins = self.data.iloc[i, j]
                loglik += wins * log(self._calculate_prob(params[i], params[j]))

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
                score_i += wins - (wins + losses) * self._calculate_prob(params[i], params[j])

            score[i] = score_i

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()

    def get_param(self, team: str) -> float:
        """Get a model parameter. For the VANBT model, this is just the team strengths.

        Args:
            team (str): Name of team. Raises an error if the team name is invalid.

        Returns:
            float: Parameter value.
        """
        self._check_fitted()
        
        try:
            index = self.teams.index(team)
        except ValueError:
            print(f"Team '{team}' not found in the model.")
            raise

        return self.params[index]

    def get_n_params(self) -> int:
        return self.n_teams
    
    def summary(self) -> str:
        return super().summary()
    
    @staticmethod
    def _calculate_odds(i: float, j: float) -> float:
        return i - j

    def get_odds(self, i: str | int, j: str | int) -> float:
        param_i = self.params[i] if type(i) == int else self.get_param(i)
        param_j = self.params[j] if type(j) == int else self.get_param(j)
        return self._calculate_odds(param_i, param_j)
    
    @staticmethod
    def _calculate_prob(i: float, j: float) -> float:
        return expit(VANBT._calculate_odds(i, j))

    def get_prob(self, i: str | int, j: str | int) -> float:
        # TODO refactor out?
        return expit(self.get_odds(i, j))


class CHABT(BaseBradleyTerry):
    """Class for the common home-ground advantage (CHA) Bradley-Terry model. The model supposes 
    there is some constant home advantage term.

    The parameter vector contains the team strengths, and the common HFA parameter at the end.
    """
    def _set_data(self, data: dict[str: DataFrame]):
        # TODO data validation
        # TODO allow for non-pandas data matrix types, ie, generic iterables
        # Assume dict of DFs for now
        assert isinstance(data, dict)
        assert list(data.keys()) == ["home", "away", "neutral"]
        assert all([i.shape[0] == i.shape[1] for i in data.values()])
        
        self.data = data
        self.teams = data["home"].columns.tolist()
        self.n_teams = len(self.teams)
        self.n_params = self.get_n_params()

    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                    continue

                home_wins = self.data["home"].iloc[i, j]
                away_wins = self.data["away"].iloc[i, j]
                neut_wins = self.data["neutral"].iloc[i, j]
                i_beats_j_home = self._calculate_prob(params[i], params[j], params[-1])
                i_beats_j_away = self._calculate_prob(params[i], params[j], -params[-1])
                i_beats_j_neut = self._calculate_prob(params[i], params[j], 0)

                loglik += home_wins * log(i_beats_j_home) + away_wins * log(i_beats_j_away) + neut_wins * log(i_beats_j_neut)

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        # Thetas
        for i in range(self.n_teams):
            score_i = 0
            for j in range(self.n_teams):
                if i == j:
                    continue

                home = self.data["home"]
                away = self.data["away"]
                neut = self.data["neutral"]
                i_beats_j_home = self._calculate_prob(params[i], params[j], params[-1])
                i_beats_j_away = self._calculate_prob(params[i], params[j], -params[-1])
                i_beats_j_neut = self._calculate_prob(params[i], params[j], 0)

                score_i += home.iloc[i, j] + away.iloc[i, j] + neut.iloc[i, j] \
                    - (home.iloc[i, j] + away.iloc[j, i]) * i_beats_j_home \
                    - (home.iloc[j, i] + away.iloc[i, j]) * i_beats_j_away \
                    - (neut.iloc[i, j] + neut.iloc[j, i]) * i_beats_j_neut

            score[i] = score_i

        # Alpha
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                j_beats_i_home = self._calculate_prob(params[j], params[i], params[-1])
                j_beats_i_away = self._calculate_prob(params[j], params[i], -params[-1])
                score[-1] += self.data["home"].iloc[i, j] * j_beats_i_away - self.data["away"].iloc[i, j] * j_beats_i_home

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def get_param(self, param_type: str, team: str = None) -> float:
        """Gets a model parameter. For the CHABT model, this is either a team strength or the
        constant home-ground advantage term.

        Args:
            param_type (str): `"strength"` for team strength; `"hga"` for home-ground advantage.
            team (str, optional): If wanting a team strength, name of the team. Defaults to None.

        Returns:
            float: Parameter value.
        """
        self._check_fitted()
        
        if param_type == "strength":
            try:
                index = self.teams.index(team)
            except ValueError:
                print(f"Team '{team}' not found in the model.")
                raise
        elif param_type == "hga":
            index = -1
        else:
            raise ValueError(f"Parameter type {param_type} is invalid. See docstring.")

        return self.params[index]
    
    def get_n_params(self):
        return self.n_teams + 1
    
    def summary(self):
        return super().summary()

    @staticmethod
    def _calculate_odds(i: float, j: float, h: float) -> float:
        return i - j + h

    def get_odds(self, i: str | int, j: str | int, venue: str = "home") -> float:
        venue_map = {"home": 1, "neutral": 0, "away": -1}
        param_i = self.params[i] if type(i) == int else self.get_param(i)
        param_j = self.params[j] if type(j) == int else self.get_param(j)
        hga = self.params[-1] * venue_map[venue]
        return self._calculate_odds(param_i, param_j, hga)
    
    @staticmethod
    def _calculate_prob(i: float, j: float, h: float) -> float:
        return expit(CHABT._calculate_odds(i, j, h))

    def get_prob(self, i: str | int, j: str | int, venue: str = "home") -> float:
        # TODO refactor out?
        return expit(self.get_odds(i, j, venue))


class CHIBT(BaseBradleyTerry):
    pass


class TSHBT(BaseBradleyTerry):
    pass


class HIEBT(BaseBradleyTerry):
    pass
