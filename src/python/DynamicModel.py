import numpy as np
from numpy import ndarray, exp, log
from pandas import DataFrame
from scipy.special import expit

from BaseModel import BaseBradleyTerry, BaseHierarchicalBT

# TODO handle teams coming in and out of data each year -> requirements for mats, etc
# TODO look into sparse matrices for data (and justify)
class VANBT(BaseBradleyTerry):
    """Class for a dyanmic 'vanilla' (VAN) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T)]`
    """
    venue_keys = None

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

                    loglik += wins * log(self._calculate_prob(theta_it, theta_jt, 0))

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

                    score_it += wins - (wins + losses) * self._calculate_prob(theta_it, theta_jt, 0)

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
                    pi_ij = self._calculate_prob(theta_it, theta_jt, 0)

                    # Off-diag elements
                    h_it_jt = (wins + losses) * pi_ij * (1 - pi_ij)
                    hess[self._get_strength_idx(i, t), self._get_strength_idx(j, t)] = h_it_jt

                    # Diag elements are negated sum of off-diag for each row...
                    h_it_diag += h_it_jt
                hess[self._get_strength_idx(i, t), self._get_strength_idx(i, t)] = -h_it_diag

        return -hess

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times
    
    def _get_hga_param(self, team=None, level=None):
        raise ValueError("VANBT model has no home-ground advantage parameters.")
    
    def summary(self) -> str:
        return super().summary()
    
    def _get_venue_map(self, i, j):
        return {"home": 0, "away": 0, "neutral": 0}


class CHABT(BaseBradleyTerry):
    """Class for a dynamic common home-ground advantage (CHA) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.
    - Common HGA `alpha`.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T), alpha]`
    """
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
    
    def _get_hga_idx(self):
        return -1
    
    def _get_hga_param(self, team=None, level=None):
        return self.params[self._get_hga_idx()]

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times + 1
    
    def summary(self) -> str:
        return super().summary()

    def _get_venue_map(self, i, j):
        return {"home": self.get_param("hga"), "neutral": 0, "away": -self.get_param("hga")}


class TSABT(BaseBradleyTerry):
    """Class for a dynamic team-specific (TS) home-ground advantage (A) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.
    - Team-specific HGA `alpha_i` for each team.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T), a_1, ..., a_I]`
    """
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
    
    def _get_hga_idx(self, i) -> int:
        return self.n_teams * self.n_times + i
    
    def _get_hga_param(self, team=None, level=None):
        try:
            team_idx = self.teams.index(team)
            return self.params[self._get_hga_idx(team_idx)]
        except ValueError:
            raise ValueError(f"Team '{team}' not found in the model.")

    def get_n_params(self) -> int:
        return self.n_teams * (self.n_times + 1)
    
    def summary(self) -> str:
        return super().summary()

    def _get_venue_map(self, i, j):
        return {"home": self.get_param("hga", i), "neutral": 0, "away": -self.get_param("hga", j)}


class CHIBT(BaseHierarchicalBT):
    """Class for a dynamic common hierarchical home-ground advantage (CHI) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.
    - Common HGA `alpha_k` for each level `k=1,...,K` of the hierarchy.

    Parameter vectorisation:
    `[(i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T), alpha_1, ..., alpha_K]`
    """
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
                    # TODO add fn to get index from level name
                    hga = params[self._get_hga_idx(int(self.rel_mat.iloc[i, j]))]
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
                    hga = params[self._get_hga_idx(int(self.rel_mat.iloc[i, j]))]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    score_it += home_wins + away_wins + neut_wins \
                    - (home_wins + away_loss) * i_beats_j_home \
                    - (away_wins + home_loss) * i_beats_j_away \
                    - (neut_wins + neut_loss) * i_beats_j_neut

                score[self._get_strength_idx(i, t)] = score_it

        # HGA terms
        for k, level in enumerate(self.levels):
            for t, year in enumerate(self.times):
                cur_year = self.data[year]
                for i in range(self.n_teams):
                    for j in range(self.n_teams):
                        if i == j: continue
                        if self.rel_mat.iloc[i, j] != level: continue

                        home_wins = cur_year["home"].iloc[i, j]
                        away_wins = cur_year["away"].iloc[i, j]
                        theta_it = params[self._get_strength_idx(i, t)]
                        theta_jt = params[self._get_strength_idx(j, t)]
                        hga = params[self._get_hga_idx(k)]
                        i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                        i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)

                        score[self._get_hga_idx(k)] += home_wins * (1 - i_beats_j_home) - away_wins * (1 - i_beats_j_away)

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
                    hga = params[self._get_hga_idx(int(self.rel_mat.iloc[i, j]))]
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

        # HGA terms
        for k, level in enumerate(self.levels):
            dummy = 0
            for t, year in enumerate(self.times):
                cur_year = self.data[year]
                for i in range(self.n_teams):
                    for j in range(self.n_teams):
                        if i == j: continue
                        if self.rel_mat.iloc[i, j] != level: continue

                        home_wins = cur_year["home"].iloc[i, j]
                        away_wins = cur_year["away"].iloc[i, j]

                        theta_it = params[self._get_strength_idx(i, t)]
                        theta_jt = params[self._get_strength_idx(j, t)]
                        hga = params[self._get_hga_idx(k)]
                        i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                        i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)

                        dummy += home_wins * i_beats_j_home * (1-i_beats_j_home) + away_wins * i_beats_j_away * (1-i_beats_j_away)
            hess[self._get_hga_idx(k), self._get_hga_idx(k)] = -dummy

        # Cross terms
        for k, level in enumerate(self.levels):
            for t, year in enumerate(self.times):
                cur_year = self.data[year]
                for i in range(self.n_teams):
                    dummy = 0
                    for j in range(self.n_teams):
                        if i == j: continue
                        if self.rel_mat.iloc[i, j] != level: continue

                        home_wins = cur_year["home"].iloc[i, j]
                        home_loss = cur_year["home"].iloc[j, i]
                        away_wins = cur_year["away"].iloc[i, j]
                        away_loss = cur_year["away"].iloc[j, i]

                        theta_it = params[self._get_strength_idx(i, t)]
                        theta_jt = params[self._get_strength_idx(j, t)]
                        hga = params[self._get_hga_idx(k)]
                        i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga)
                        i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga)

                        dummy += -(home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home) \
                        + (away_wins + home_loss) * i_beats_j_away * (1-i_beats_j_away)
                    hess[self._get_strength_idx(i, t), self._get_hga_idx(k)] = hess[self._get_hga_idx(k), self._get_strength_idx(i, t)] = dummy

        return -hess
    
    def _get_hga_idx(self, k: int):
        return self.n_teams * self.n_times + k
    
    def _get_hga_param(self, team=None, level=None):
        try:
            level_idx = self.levels.index(level)
            return self.params[self._get_hga_idx(level_idx)]
        except ValueError:
            raise ValueError(f"Level '{level}' not found in the model.")

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times + self.n_levels
    
    def summary(self) -> str:
        return super().summary()
    
    def _get_venue_map(self, i, j):
        k = self.rel_mat.loc[i, j]
        return {"home": self._get_hga_idx(k), "neutral": 0, "away": -self._get_hga_idx(k)}


class TSIBT(BaseHierarchicalBT):
    """Class for a dyanmic team-specific hierarchical home-ground advantage (TSI) Bradley-Terry model.

    Parameters:
    - Team strengths `theta_{it}` for each team `i=1, ..., I` in each time block `t=1,...,T`.
    - Team-specific HGA `alpha_{ik}` for each level `k=1,...,K` of the hierarchy.

    Parameter vectorisation:
    `[
        (i=1, t=1), (i=2, t=1), ..., (i=I, t=1), (i=1, t=2), ..., (i=I, t=T), 
        alpha_11, ..., alpha_I1, alpha_12, ..., alpha_I2, ..., alpha_1K, ..., alpha_IK
    ]`
    """
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
                    # TODO add fn to get index from level name
                    r_ij = int(self.rel_mat.iloc[i, j])
                    hga_i = params[self._get_hga_idx(i, r_ij)]
                    hga_j = params[self._get_hga_idx(j, r_ij)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga_j)
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
                    r_ij = int(self.rel_mat.iloc[i, j])
                    hga_i = params[self._get_hga_idx(i, r_ij)]
                    hga_j = params[self._get_hga_idx(j, r_ij)]
                    i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                    i_beats_j_away = self._calculate_prob(theta_it, theta_jt, -hga_j)
                    i_beats_j_neut = self._calculate_prob(theta_it, theta_jt, 0)

                    score_it += home_wins + away_wins + neut_wins \
                    - (home_wins + away_loss) * i_beats_j_home \
                    - (away_wins + home_loss) * i_beats_j_away \
                    - (neut_wins + neut_loss) * i_beats_j_neut

                score[self._get_strength_idx(i, t)] = score_it

        # HGA terms
        for k, level in enumerate(self.levels):
            for i in range(self.n_teams):
                score_hi = 0
                for t, year in enumerate(self.times):
                    cur_year = self.data[year]
                    for j in range(self.n_teams):
                        if i == j: continue
                        if self.rel_mat.iloc[i, j] != level: continue

                        home_wins = cur_year["home"].iloc[i, j]
                        away_loss = cur_year["away"].iloc[j, i]
                        theta_it = params[self._get_strength_idx(i, t)]
                        theta_jt = params[self._get_strength_idx(j, t)]
                        hga_i = params[self._get_hga_idx(i, k)]
                        i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)
                        
                        score_hi += home_wins - (home_wins + away_loss) * i_beats_j_home

                score[self._get_hga_idx(i, k)] = score_hi

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
                    r_ij = int(self.rel_mat.iloc[i, j])
                    hga_i = params[self._get_hga_idx(i, r_ij)]
                    hga_j = params[self._get_hga_idx(j, r_ij)]
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

        # HGA terms
        for k, level in enumerate(self.levels):
            for i in range(self.n_teams):
                dummy = 0
                for t, year in enumerate(self.times):
                    cur_year = self.data[year]
                    for j in range(self.n_teams):
                        if i == j: continue
                        if self.rel_mat.iloc[i, j] != level: continue

                        home_wins = cur_year["home"].iloc[i, j]
                        away_loss = cur_year["away"].iloc[j, i]

                        theta_it = params[self._get_strength_idx(i, t)]
                        theta_jt = params[self._get_strength_idx(j, t)]
                        hga_i = params[self._get_hga_idx(i, k)]
                        i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)

                        dummy += (home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home)
                hess[self._get_hga_idx(i, k), self._get_hga_idx(i, k)] = -dummy

        # Cross terms
        for k, level in enumerate(self.levels):
            for i in range(self.n_teams):
                for t, year in enumerate(self.times):
                    cur_year = self.data[year]
                    dummy = 0
                    for j in range(self.n_teams):
                        if i == j: continue
                        if self.rel_mat.iloc[i, j] != level: continue

                        home_wins = cur_year["home"].iloc[i, j]
                        home_loss = cur_year["home"].iloc[j, i]
                        away_wins = cur_year["away"].iloc[i, j]
                        away_loss = cur_year["away"].iloc[j, i]

                        theta_it = params[self._get_strength_idx(i, t)]
                        theta_jt = params[self._get_strength_idx(j, t)]
                        hga_i = params[self._get_hga_idx(i, k)]
                        i_beats_j_home = self._calculate_prob(theta_it, theta_jt, hga_i)

                        dummy_2 = (home_wins + away_loss) * i_beats_j_home * (1-i_beats_j_home)
                        dummy += dummy_2
                        hess[self._get_strength_idx(j, t), self._get_hga_idx(i, k)] = hess[self._get_hga_idx(i, k), self._get_strength_idx(j, t)] = dummy_2
                    hess[self._get_strength_idx(i, t), self._get_hga_idx(i, k)] = hess[self._get_hga_idx(i, k), self._get_strength_idx(i, t)] = -dummy

        return -hess

    def _get_hga_idx(self, i: int, k: int):
        return self.n_teams * self.n_times + i + self.n_teams * k
    
    def _get_hga_param(self, team=None, level=None):
        try:
            team_idx = self.teams.index(team)
            level_idx = self.levels.index(level)
            return self.params[self._get_hga_idx(team_idx, level_idx)]
        except ValueError:
            raise ValueError(f"Team '{team}' or level '{level}' not found in the model.")

    def get_n_params(self) -> int:
        return self.n_teams * self.n_times + self.n_teams * self.n_levels
    
    def summary(self) -> str:
        return super().summary()

    def _get_venue_map(self, i, j):
        k = self.rel_mat.loc[i, j]
        return {"home": self.get_param("hga", team=i, level=k), "neutral": 0, "away": -self.get_param("hga", team=j, level=k)}
