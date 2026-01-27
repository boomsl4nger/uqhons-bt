import numpy as np
from numpy import ndarray, exp, log
from pandas import DataFrame
from scipy.special import expit

from BaseModel import BaseBradleyTerry, BaseHierarchicalBT

# TODO handle teams coming in and out of data each year -> requirements for mats, etc
# TODO change name to VANBT and remove static class(es)
class DyVANBT(BaseBradleyTerry):
    """Class for a dyanmic 'vanilla' (VAN) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T)]`
    """
    def _set_data(self, data: dict[int: DataFrame]):
        # TODO docstring
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
        # TODO think about using team names w/ .loc? is this handled enough in _set_data? what about the times then (needed for dict)?
        # TODO vectorisation in an attempt to speed these LLH fns up for fitting...
        # Precompute probabilities as a matrix or something?
        loglik = 0
        for t, year in enumerate(self.times):
            cur_mat = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: # Assume teams to not play themselves (main diagonal has zeroes)
                        continue

                    wins = cur_mat.iloc[i, j]
                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]

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
                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]

                    score_it += wins - (wins + losses) * self._calculate_prob(theta_it, theta_jt)

                score[self._get_strength_idx(i, t)] = score_it

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        # TODO can we save time by using the Hessians' symmetry?
        # TODO also vectorise ops for speed, try to combine for loops if run time is too long...
        hess = np.zeros((self.n_params, self.n_params))
        for t, year in enumerate(self.times):
            cur_mat = self.data[year]
            for i in range(self.n_teams):
                h_it_diag = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    wins = cur_mat.iloc[i, j]
                    losses = cur_mat.iloc[j, i]
                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    pi_ij = self._calculate_prob(theta_it, theta_jt)

                    # Off-diag elements
                    h_it_jt = (wins + losses) * pi_ij * (1 - pi_ij)
                    hess[self._get_strength_idx(i, t), self._get_strength_idx(j, t)] = h_it_jt

                    # Diag elements are negated sum of off-diag for each row...
                    h_it_diag += h_it_jt
                hess[self._get_strength_idx(i, t), self._get_strength_idx(i, t)] = -h_it_diag

        return -hess

    def get_param(self, team: str, time: str | int) -> float:
        """Get a model parameter. The VANBT model only has team strengths in each time block.

        Args:
            team (str): Name of team. Raises an error if invalid.
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

        return self.params[self._get_strength_idx(team_idx, time_idx)]

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times
    
    def summary(self) -> str:
        return super().summary()
    
    @staticmethod
    def _calculate_odds(it: float, jt: float) -> float:
        return it - jt

    def get_odds(self, i: str, j: str, t: str | int) -> float:
        return self._calculate_odds(self.get_param(i, t), self.get_param(j, t))
    
    @staticmethod
    def _calculate_prob(it: float, jt: float) -> float:
        return expit(DyVANBT._calculate_odds(it, jt))

    def get_prob(self, i: str | int, j: str | int, t: str | int) -> float:
        return expit(self.get_odds(i, j, t))


class DyCHABT(BaseBradleyTerry):
    """Class for a dyanmic common home-ground advantage (CHA) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.
    - Common HGA `alpha`.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T), alpha]`
    """
    def _set_data(self, data: dict[int: DataFrame]):
        # TODO data validation function, check each matrix is right etc, check times make sense, better checking for names all same
        # TODO allow for non-pandas data matrix types, ie, generic iterables
        # Expecting: dict {year: {venue_type: win_matrix}}
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a square DataFrame or dict of such.")
        
        self.data = data
        self.teams = list(data.values())[0]["home"].columns.tolist()
        self.n_teams = len(self.teams)
        self.times = sorted(list(data.keys()))
        self.n_times = len(self.times)
        self.n_params = self.get_n_params()

    def _log_likelihood(self, params: ndarray) -> float:
        loglik = 0
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: continue # Assume teams don't play themselves

                    home_wins = cur_year["home"].iloc[i, j]
                    away_wins = cur_year["away"].iloc[i, j]
                    neut_wins = cur_year["neutral"].iloc[i, j]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga = params[self._get_hga_idx()]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    loglik += home_wins * log(i_beats_j_home) + away_wins * log(i_beats_j_away) + neut_wins * log(i_beats_j_neut)

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                score_it = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    home_loss = cur_year["home"].iloc[j, i]
                    away_wins = cur_year["away"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]
                    neut_wins = cur_year["neutral"].iloc[i, j]
                    neut_loss = cur_year["neutral"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga = params[self._get_hga_idx()]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    score_it += home_wins + away_wins + neut_wins \
                    - (home_wins + away_loss) * i_beats_j_home \
                    - (away_wins + home_loss) * i_beats_j_away \
                    - (neut_wins + neut_loss) * i_beats_j_neut

                score[self._get_strength_idx(i, t)] = score_it

        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    away_wins = cur_year["away"].iloc[i, j]
                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga = params[self._get_hga_idx()]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)

                    score[self._get_hga_idx()] += home_wins * (1 - i_beats_j_home) - away_wins * (1 - i_beats_j_away)

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        hess = np.zeros((self.n_params, self.n_params))

        # Team strengths block
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                h_it_diag = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    home_loss = cur_year["home"].iloc[j, i]
                    away_wins = cur_year["away"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]
                    neut_wins = cur_year["neutral"].iloc[i, j]
                    neut_loss = cur_year["neutral"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga = params[self._get_hga_idx()]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    # Off-diags
                    h_it_jt = (home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home) \
                    + (away_wins + home_loss) * i_beats_j_away * (1-i_beats_j_away) \
                    + (neut_wins + neut_loss) * i_beats_j_neut * (1-i_beats_j_neut)
                    hess[self._get_strength_idx(i, t), self._get_strength_idx(j, t)] = h_it_jt

                    # Diag elements are negated sum of off-diag for each row...
                    h_it_diag += h_it_jt
                hess[self._get_strength_idx(i, t), self._get_strength_idx(i, t)] = -h_it_diag

        # HGA term
        dummy = 0
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    away_wins = cur_year["away"].iloc[i, j]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga = params[self._get_hga_idx()]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)

                    dummy += home_wins * i_beats_j_home * (1-i_beats_j_home) + away_wins * i_beats_j_away * (1-i_beats_j_away)
        hess[self._get_hga_idx(), self._get_hga_idx()] = -dummy

        # Cross terms
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                dummy = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    home_loss = cur_year["home"].iloc[j, i]
                    away_wins = cur_year["away"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga = params[self._get_hga_idx()]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)

                    dummy += -(home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home) \
                    + (away_wins + home_loss) * i_beats_j_away * (1-i_beats_j_away)
                hess[self._get_strength_idx(i, t), self._get_hga_idx()] = hess[self._get_hga_idx(), self._get_strength_idx(i, t)] = dummy

        return -hess

    def get_param(self, param_type: str, team: str = None, time: str | int = None) -> float:
        # TODO param_type should be one of [hga, strength]
        """Get a model parameter. The CHABT model has team strengths in each time block and a common
        home-ground advantage effect.

        Args:
            param_type (str): `"strength"` for team strength; `"hga"` for home-ground advantage.
            team (str, optional): Name of team. Default to None.
            time (str | int, optional): Name of time block. Default to None.

        Returns:
            float: Parameter value.
        """
        self._check_fitted()

        if param_type == "strength":
            try:
                team_idx = self.teams.index(team)
                time_idx = self.times.index(time)
                index = self._get_strength_idx(team_idx, time_idx)
            except ValueError:
                print(f"Team '{team}' or time '{time}' not found in the model.")
                raise
        elif param_type == "hga":
            index = self._get_hga_idx()
        else:
            raise ValueError(f"Parameter type {param_type} is invalid. See docstring.")

        return self.params[index]
    
    def _get_hga_idx(self):
        return -1

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times + 1
    
    def summary(self) -> str:
        return super().summary()
    
    @staticmethod
    def _calculate_odds(it: float, jt: float, h: float) -> float:
        return it - jt + h

    def get_odds(self, i: str, j: str, t: str | int, venue: str = "home") -> float:
        # TODO venue must be one of [home, away, neutral]
        venue_map = {"home": 1, "neutral": 0, "away": -1}
        hga = self._get_hga_idx() * venue_map[venue]
        return self._calculate_odds(self.get_param(i, t), self.get_param(j, t), hga)
    
    @staticmethod
    def _calculate_prob(it: float, jt: float, h: float) -> float:
        return expit(DyCHABT._calculate_odds(it, jt, h))

    def get_prob(self, i: str, j: str, t: str | int, venue: str = "home") -> float:
        return expit(self.get_odds(i, j, t, venue))


class DyTSABT(BaseBradleyTerry):
    """Class for a dyanmic team-specific (TS) home-ground advantage (A) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.
    - Team-specific HGA `alpha_i` for each team.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T), a_1, ..., a_I]`
    """
    def _set_data(self, data: dict[int: DataFrame]):
        # TODO data validation function, check each matrix is right etc, check times make sense, better checking for names all same
        # TODO allow for non-pandas data matrix types, ie, generic iterables
        # Expecting: dict {year: {venue_type: win_matrix}}
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a square DataFrame or dict of such.")
        
        self.data = data
        self.teams = list(data.values())[0]["home"].columns.tolist()
        self.n_teams = len(self.teams)
        self.times = sorted(list(data.keys()))
        self.n_times = len(self.times)
        self.n_params = self.get_n_params()

    def _log_likelihood(self, params: ndarray) -> float:
        # TODO just call the chabt llh fn but ensure that the prob calcs are called correctly
        loglik = 0
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i == j: continue # Assume teams don't play themselves

                    home_wins = cur_year["home"].iloc[i, j]
                    away_wins = cur_year["away"].iloc[i, j]
                    neut_wins = cur_year["neutral"].iloc[i, j]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga_i = params[self._get_hga_idx(i)]
                    hga_j = params[self._get_hga_idx(j)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga_j)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    loglik += home_wins * log(i_beats_j_home) + away_wins * log(i_beats_j_away) + neut_wins * log(i_beats_j_neut)

        return -loglik

    def _score(self, params: ndarray) -> ndarray:
        score = np.zeros(self.n_params)
        # TODO reusing element calcs for score and hess from chabt
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                score_it = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    home_loss = cur_year["home"].iloc[j, i]
                    away_wins = cur_year["away"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]
                    neut_wins = cur_year["neutral"].iloc[i, j]
                    neut_loss = cur_year["neutral"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga_i = params[self._get_hga_idx(i)]
                    hga_j = params[self._get_hga_idx(j)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga_j)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    score_it += home_wins + away_wins + neut_wins \
                    - (home_wins + away_loss) * i_beats_j_home \
                    - (away_wins + home_loss) * i_beats_j_away \
                    - (neut_wins + neut_loss) * i_beats_j_neut

                score[self._get_strength_idx(i, t)] = score_it

        for i in range(self.n_teams):
            score_hi = 0
            for t, year in enumerate(self.times):
                cur_year = self.data[year]
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]
                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga_i = params[self._get_hga_idx(i)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                    
                    score_hi += home_wins - (home_wins + away_loss) * i_beats_j_home

            score[self._get_hga_idx(i)] = score_hi

        return -score

    def _hessian(self, params: ndarray) -> ndarray:
        hess = np.zeros((self.n_params, self.n_params))

        # Team strengths block
        for t, year in enumerate(self.times):
            cur_year = self.data[year]
            for i in range(self.n_teams):
                h_it_diag = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    home_loss = cur_year["home"].iloc[j, i]
                    away_wins = cur_year["away"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]
                    neut_wins = cur_year["neutral"].iloc[i, j]
                    neut_loss = cur_year["neutral"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga_i = params[self._get_hga_idx(i)]
                    hga_j = params[self._get_hga_idx(j)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga_j)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    # Off-diags
                    h_it_jt = (home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home) \
                    + (away_wins + home_loss) * i_beats_j_away * (1-i_beats_j_away) \
                    + (neut_wins + neut_loss) * i_beats_j_neut * (1-i_beats_j_neut)
                    hess[self._get_strength_idx(i, t), self._get_strength_idx(j, t)] = h_it_jt

                    # Diag elements are negated sum of off-diag for each row...
                    h_it_diag += h_it_jt
                hess[self._get_strength_idx(i, t), self._get_strength_idx(i, t)] = -h_it_diag

        # HGA term
        for i in range(self.n_teams):
            dummy = 0
            for t, year in enumerate(self.times):
                cur_year = self.data[year]
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga_i = params[self._get_hga_idx(i)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)

                    dummy += (home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home)
            hess[self._get_hga_idx(i), self._get_hga_idx(i)] = -dummy

        # Cross terms
        for i in range(self.n_teams):
            for t, year in enumerate(self.times):
                cur_year = self.data[year]
                dummy = 0
                for j in range(self.n_teams):
                    if i == j: continue

                    home_wins = cur_year["home"].iloc[i, j]
                    home_loss = cur_year["home"].iloc[j, i]
                    away_wins = cur_year["away"].iloc[i, j]
                    away_loss = cur_year["away"].iloc[j, i]

                    theta_it = params[self._get_strength_idx(i, t)]
                    theta_jt = params[self._get_strength_idx(j, t)]
                    hga_i = params[self._get_hga_idx(i)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)

                    dummy_2 = (home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home)
                    dummy += dummy_2
                    hess[self._get_strength_idx(j, t), self._get_hga_idx(i)] = hess[self._get_hga_idx(i), self._get_strength_idx(j, t)] = dummy_2
                hess[self._get_strength_idx(i, t), self._get_hga_idx(i)] = hess[self._get_hga_idx(i), self._get_strength_idx(i, t)] = -dummy

        return -hess

    def get_param(self, param_type: str, team: str, time: str | int = None) -> float:
        # TODO param_type should be one of [hga, strength] and no default
        """Get a model parameter. In the TSABT model, each team has strengths for each time block 
        and a constant home-ground advantage.

        Args:
            param_type (str): `"strength"` for team strength; `"hga"` for home-ground advantage.
            team (str): Name of team.
            time (str | int, optional): Name of time block. Default to None.

        Returns:
            float: Parameter value.
        """
        self._check_fitted()

        if param_type == "strength":
            try:
                team_idx = self.teams.index(team)
                time_idx = self.times.index(time)
                index = self._get_strength_idx(team_idx, time_idx)
            except ValueError:
                print(f"Team '{team}' or time '{time}' not found in the model.")
        elif param_type == "hga":
            try:
                team_idx = self.teams.index(team)
                index = self._get_hga_idx(team_idx)
            except ValueError:
                print(f"Team '{team}' not found in the model.")
        else:
            raise ValueError(f"Parameter type {param_type} is invalid. See docstring.")

        return self.params[index]
    
    def _get_hga_idx(self, i) -> int:
        return self.n_teams * self.n_times + i

    def get_n_params(self) -> int:
        return self.n_teams * (self.n_times + 1)
    
    def summary(self) -> str:
        return super().summary()
    
    @staticmethod
    def _calculate_odds(it: float, jt: float, h: float) -> float:
        return it - jt + h

    def get_odds(self, i: str, j: str, t: str | int, venue: str = "home") -> float:
        venue_map = {"home": self.get_param("hga", i), "neutral": 0, "away": -self.get_param("hga", j)}
        hga = venue_map[venue]
        return self._calculate_odds(self.get_param(i, t), self.get_param(j, t), hga)
    
    @staticmethod
    def _calculate_prob(it: float, jt: float, h: float) -> float:
        return expit(DyTSABT._calculate_odds(it, jt, h))

    def get_prob(self, i: str, j: str, t: str | int, venue: str = "home") -> float:
        return expit(self.get_odds(i, j, t, venue))
