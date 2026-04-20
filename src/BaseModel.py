from collections import defaultdict
import networkx as nx
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize
from scipy.special import expit
from typing import Literal

from hypothesis_tests import model_statistics

# TODO make abstract base class
# TODO consider staticmethods (LLH, odds, prob)
# TODO naming for odds (when it's actually log-odds)
# TODO type hint fields? eg self.data: dict, self.teams: list
# TODO consider making a wrapper for fully dynamic models

class BaseBradleyTerry():
    """Abstract base class for our Bradley-Terry models. Mainly specifies the methods that each 
    model will need to implement, such as the log-likelihood function.
    """
    venue_keys = {"home", "away", "neutral"}

    ## ===== INIT FNS ===== ##

    def __init__(self):
        """Initialise a Bradley-Terry model.
        
        Setting should be done before model fitting with `_set_data()` method.
        """
        # Dataset characteristics
        self.data = None
        self.n_obs = 0
        self.teams = None
        self.n_teams = 0
        self.times = None
        self.n_times = 0

        # Constraints and edge case params
        self._constraint_team_idx = 0
        self.inactive_teams = defaultdict(list)
        self._inactive_idxs = None
        self.global_teams = None

        # Model fit items
        self.params = None
        self.n_params = 0
        self.n_params_active = 0
        self._rel_mat = None
        self._fit_summary = None
        self._hess_inv = None
        self.errors = None

    def _set_data(self, data: dict):
        """Validate and set data-related parameters.

        Expected data structure is a dictionary with time block names as keys and win matrices as values.
        Currently expecting the win matrices to be pandas DataFrames with team names as column and index names.
        If using a HGA, we expect a nested dict with the venue types `"home", "away", "neutral"`.
        - Vanilla model: `{time: DataFrame}`
        - HGA models: `{time: {venue: DataFrame}}`

        Args:
            data (dict): Input competition data. Format depends on model type (see above).
        """
        self._validate_data(data)
        self.data = data
        self._finalise_params()

    def _validate_data(self, data: dict):
        """Validate a given data structure of win matrices. See `_set_data`."""
        # TODO think about allowing non-pandas win matrix types (eg 2D numpy arrays)
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a dictionary keyed by time blocks.")

        # Extract time blocks
        self.times = sorted(data)
        self.n_times = len(self.times)

        # It's difficult to write a data validation function when the data requirements for the 
        # children are different (VAN is just dict of dfs, while rest are dict of dict of dfs)
        # We know that data is at least a dict and the child classes have the `venue_keys` set to
        # distinguish between VAN or not, so check first item and go from there
        first = data[self.times[0]]
        if self.venue_keys is None: # Vanilla model
            if not isinstance(first, DataFrame):
                raise ValueError(f"{self.__class__.__name__} requires {{time: DataFrame}} structure.")
            ref_df = first
        else:                       # HGA models
            if not (isinstance(first, dict) and "home" in first and isinstance(first["home"], DataFrame)):
                raise ValueError(f"{self.__class__.__name__} requires "f"{{time: {{venue: DataFrame}}}} structure.")
            ref_df = first["home"]

        self.teams = list(ref_df.columns)
        self.n_teams = len(self.teams)

        def iter_matrices(block, t):
            if self.venue_keys is None:
                return (block,)
            if set(block) != self.venue_keys:
                raise ValueError(f"Time {t} must contain keys {self.venue_keys}, got {set(block)}")
            return block.values()
        
        def find_inactive_teams(mat: DataFrame) -> list[str]:
            row_sums = mat.sum(axis=1)
            col_sums = mat.sum(axis=0)
            return mat.index[(row_sums == 0) & (col_sums == 0)].tolist()
        
        def is_strongly_connected(mat: DataFrame) -> bool:
            G = nx.DiGraph()

            for i in mat.index:
                for j in mat.columns:
                    if i != j and mat.loc[i, j] > 0:
                        G.add_edge(i, j)

            return nx.is_strongly_connected(G)

        # Main data validation loop
        for t in self.times:
            iter_mats = iter_matrices(data[t], t)
            for df in iter_mats:
                # Check is DataFrame
                if not isinstance(df, DataFrame):
                    raise ValueError(f"Matrix in time block {t} is not a DataFrame.")

                # Check square matrix
                if df.shape[0] != df.shape[1]:
                    raise ValueError(f"Matrix in time block {t} is not square: {df.shape}")
                
                # Check team names are all equal across dfs
                if not (df.columns.equals(ref_df.columns) and df.index.equals(ref_df.index)):
                    raise ValueError(f"Team mismatch in time block {t}.")
                
                # Check win counts are non-negative
                if (df.values < 0).any():
                    raise ValueError(f"Negative values detected in time block {t}.")
                
                self.n_obs += df.to_numpy().sum()

            win_mat = sum(iter_mats)

            # Check for inactive teams
            inactive = find_inactive_teams(win_mat)
            for team in inactive:
                self.inactive_teams[team].append(t)
            
            win_mat_active = win_mat.drop(index=inactive, columns=inactive)

            # Check there are at least two active teams
            if len(win_mat_active) < 2:
                raise ValueError(f"Matrix in time block {t} has less than two active competitors!")
            
            # Check strong connectivity of active teams
            if not is_strongly_connected(win_mat_active): # Just warn for now, could still work
                print(f"Warning: win matrix for time block {t} is NOT strongly connected. Parameter estimates may not converge.")

        # Check there is at least one global reference team
        self.global_teams = sorted(list(set(self.teams) - set(self.inactive_teams.keys())))
        if self.global_teams is None:
            raise ValueError("No global reference team could be identified.")
        
        # Finally... set the reference team to be alphabetically first global team
        self._constraint_team_idx = self.teams.index(self.global_teams[0])

    def _find_inactive_idxs(self):
        """Determine which parameter vector indices are inactive (i.e., reference team or missing)."""
        ref_team_idxs = [self._get_strength_idx(self._constraint_team_idx, t) for t in range(self.n_times)]
        inactive_team_idxs = [
            self._get_strength_idx(self.teams.index(team), self.times.index(t)) 
            for team, times in self.inactive_teams.items() 
            for t in times
        ]
        return np.unique(ref_team_idxs + inactive_team_idxs)
                
    def _finalise_params(self):
        """Default setting number of parameters."""
        self._inactive_idxs = self._find_inactive_idxs()
        self.n_params = self.get_n_params()
        n_inactive_params = len([(team, t) for team, times in self.inactive_teams.items() for t in times])
        self.n_params_active = self.n_params - self.n_times - n_inactive_params

    def _check_fitted(self):
        """Check if the model parameters have been fit, and raises an exception if not."""
        if self.params is None:
            raise RuntimeError("Model parameters are not yet fitted.")
    
    ## ===== MODEL FNS ===== ##

    def _log_likelihood(self, params: ndarray) -> float:
        """Calculates the (negative) log-likelihood for the Bradley-Terry model.

        Args:
            params (ndarray): List of model parameters.

        Returns:
            float: Value of the (negative) LLH.
        """
        raise NotImplementedError()

    def _score(self, params: ndarray) -> ndarray:
        """Calculates the gradient of the (negative) log-likelihood for the Bradley-Terry model.

        Args:
            params (ndarray): List of model parameters.

        Returns:
            ndarray: Gradient of the (negative) LLH.
        """
        raise NotImplementedError()

    def _hessian(self, params: ndarray) -> ndarray:
        """Calculates the Hessian of the (negative) log-likelihood for the Bradley-Terry model.

        Args:
            params (ndarray): List of model parameters.

        Returns:
            ndarray: Hessian of the (negative) LLH.
        """
        raise NotImplementedError()

    def fit(self, data: DataFrame, verbose: bool = False):
        """Fit the Bradley-Terry model to the data. The model parameters are estimated by minimising
        the negative log-likelihood. Currently uses the BFGS method, which requires the gradient of
        the objective function, i.e., the (negative) score statistic.

        See `_init_params()` for setting initial parameter vector.

        Args:
            data (DataFrame): Data for the model.
            verbose (bool, optional): If true, prints convergence messages from `scipy.minimize()`.
                Defaults to False.

        Returns:
            self: Fitted model.
        """
        self._set_data(data)
        return self._fit_model(verbose=verbose)

    def _fit_model(self, verbose: bool = False):
        """Fit the Bradley-Terry model to the data. This function should be called internally after
        `_set_data`. This is separated because the hierarchical models require the rel mat. In
        practice, fitting should be done via the `fit` method.
        """
        # mask for active parameters
        active_mask = np.ones(self.n_params, dtype=bool)
        active_mask[self._inactive_idxs] = False

        def expand(theta_active):
            theta_full = np.zeros(self.n_params)
            theta_full[active_mask] = theta_active
            return theta_full

        def fun(theta_active):
            return self._log_likelihood(expand(theta_active))

        def jac(theta_active):
            full_grad = self._score(expand(theta_active))
            return full_grad[active_mask]
        
        # initial guess only for active params, TODO better initialisation
        x0 = np.zeros(active_mask.sum())

        result = minimize(
            fun = fun, jac = jac, x0 = x0,
            method = "BFGS",
            options = {"disp": verbose}
        )

        self._fit_summary = result
        if result.success:
            self.params = expand(result.x)
            self.rebase_abilities()
            if verbose:
                print("Successfully fit model parameters.\n")
                print(self._fit_summary)
        else:
            print(f"Model fit error: {result.message}")

        return self
    
    def rebase_abilities(self, method: Literal["first_index", "worst", "custom"] = "first_index", custom_team: str = None):
        """Helper function to rebase the estimated team strength parameters such that a chosen team 
        has zero strength in each time block. Needed to enforce model identifiability constraint(s).
        
        Note: due to our non-parametric handling of time, the reference team must be present in each
        time block. These teams can be seen via the `global_teams` field of this object.

        Args:
            method (Literal["first_index", "worst", "custom"], optional): Method for rebasing, options are
                * "first_index" which takes the team with index 0 (default);
                * "worst" which takes the team with worst strength on average; or,
                * "custom" which lets the user specify any valid team name.
            custom_team (str, optional): If `method="custom"`, specify which team to set to zero.
        """
        self._check_fitted()

        if method == "worst":
            rankings = self.get_ranking(sort_by="Team")
            constraint_idx_tentative = np.argmin(rankings["Average"])
        elif method == "custom" and custom_team is not None:
            constraint_idx_tentative = self.teams.index(custom_team)
        else: # Assume default
            constraint_idx_tentative = self.teams.index(self.global_teams[0])

        if self.teams[constraint_idx_tentative] not in self.global_teams:
            raise ValueError(f"Team {self.teams[constraint_idx_tentative]} does not appear in every time block!")
        
        self._constraint_team_idx = constraint_idx_tentative

        # Do the rebasing
        for t in range(self.n_times):
            start_index = t * self.n_teams
            end_index = start_index + self.n_teams
            self.params[start_index:end_index] -= self.params[start_index + self._constraint_team_idx]

        # Set inactive parameters to NaN rather than zero
        inactive_idxs = [
            self._get_strength_idx(self.teams.index(team), self.times.index(t)) 
            for team, times in self.inactive_teams.items() 
            for t in times
        ]
        self.params[inactive_idxs] = np.nan

        self._inactive_idxs = self._find_inactive_idxs()

        # Check if we need to rebase errors
        if self.errors is not None:
            self._calculate_errors()

    def _calculate_errors(self):
        """Calculate the standard errors for the model parameters using the inverse of the Hessian.
        Hessian should be implemented analytically, but numerical one is available as an element of
        `self._fit_summary`."""
        self._check_fitted()
        
        # We need to consider the zeroed parameter from the identifiability constraint
        # ^ This is set during the rebase_abilities call and controlled by self.constraint_team_idx
        # Calculate the full Hessian, then remove row/col and invert (should now be non-singular)
        # Note: need to remove row/col for team in EACH time block for dynamic models!
        # Standard errors are the sqrts of diagonal elements (obs Fisher approximates covariance mat)
        # Finally, just set the zeroed parameter SE to zero itself

        params_swap_nan_zero = [i if not np.isnan(i) else 0.0 for i in self.params]
        hess = self._hessian(params_swap_nan_zero)
        hess_reduced = np.delete(np.delete(hess, self._inactive_idxs, axis=0), self._inactive_idxs, axis=1)

        try:
            self._hess_inv = np.linalg.inv(hess_reduced)
            self.errors = np.sqrt(np.maximum(np.diag(self._hess_inv), 0)) # Maximum to prevent rare nans
            for idx in sorted(self._inactive_idxs):
                self.errors = np.insert(self.errors, idx, np.nan)
        except np.linalg.LinAlgError:
            raise RuntimeError("Cannot calculate standard errors: Hessian is singular.")
    
    ## ===== GETTERS ===== ##

    def get_teams(self) -> list:
        """Get the list of team names in the data."""
        return self.teams
    
    def get_n_teams(self) -> int:
        """Get the number of teams in the data."""
        return self.n_teams
    
    def get_times(self):
        """Get the names of the time blocks in the data. Usually should be years (ints)."""
        return self.times
    
    def get_n_times(self):
        """Get the number of time blocks in the data."""
        return self.n_times
    
    def get_n_params(self) -> int:
        """Get the number of parameters in the model."""
        raise NotImplementedError()
    
    def _get_rel_mat(self) -> ndarray:
        """Get the (implicit) relationship matrix for the model. Needed for checking nestedness."""
        raise NotImplementedError()
    
    def _get_strength_idx(self, i, t) -> int:
        """Get the parameter vector index of a team strength.

        Args:
            i: Team index.
            t: Time index.
        """
        return i + self.n_teams * t
    
    def _get_hga_idx(self) -> int:
        raise NotImplementedError()

    def get_param(self, param_type: Literal["strength", "hga"] = "strength", team = None, time = None, level = None) -> float:
        """User-friendly function to get a specific parameter from the model, using names of teams 
        and times rather than indices. This function is generic across all models, i.e., if using a
        non-hierarchical model then the `level` parameter can be left blank.

        Args:
            param_type (str, optional): Use "strength" for team strengths or "hga" for home-ground advantage. Defaults to "strength".
            team (optional): Name of the team. Defaults to None.
            time (optional): Name of the time block. Defaults to None.
            level (optional): Relationship level between teams. Defaults to None.

        Returns:
            float: Estimated parameter value.
        """
        self._check_fitted()
        
        if param_type == "strength":
            try:
                team_idx = self.teams.index(team)
                time_idx = self.times.index(time)
                return self.params[self._get_strength_idx(team_idx, time_idx)]
            except ValueError:
                raise ValueError(f"Team '{team}' or time '{time}' not found in the model.")
        if param_type == "hga":
            return self._get_hga_param(team, level)
        
    def _get_hga_param(self, team = None) -> float:
        raise NotImplementedError()
    
    def get_ranking(
            self, sort_by: Literal["Team", "Time", "Average"] = "Time", sort_time: str = None,
            include_average: bool = True, include_errors: bool = False, mean_center: bool = False,
            as_ranks: bool = False
        ) -> DataFrame:
        """Returns a neat DF of the team strengths in each time block, with various sorting options.

        Args:
            sort_by (str, optional): What to sort by. "Team" sorts alphabetically; "Time" sorts by block `sort_time`. Defaults to "Time".
            sort_time (str, optional): Name of time block to sort by. Defaults to None.
            include_average (bool, optional): If True, includes a column for the average strengths. Defaults to True.
            include_errors (bool, optional): If True, includes SEs after each time column. Defaults to False.
            mean_center (bool, optional): If True, subtracts average strength for each year. Defaults to False.
            as_ranks (bool, optional): If True, replaces the strengths by their placements in each year. Defaults to False.

        Returns:
            DataFrame: Team strengths for each year.
        """
        self._check_fitted()

        # Reshape to (I x T) matrix format
        n_strength_params = self.n_times * self.n_teams
        ability_matrix = self.params[:n_strength_params].reshape(self.n_times, self.n_teams).T
        rankings = DataFrame(ability_matrix, index = self.teams, columns = self.times)

        # Optional: mean-centering
        if mean_center: rankings = rankings - rankings.mean(axis=0)

        # Optional: use placement rankings
        if as_ranks:
            rankings = rankings.rank(axis=0, ascending=False, method="average")
            include_errors = False  # SEs aren't meaningful now
        
        # Calculate averages for sorting
        avgs = rankings.mean(axis=1)
        if include_average: rankings["Average"] = avgs

        # Optional: add standard errors
        if include_errors:
            if self.errors is None: self._calculate_errors()
            error_matrix = self.errors[:n_strength_params].reshape(self.n_times, self.n_teams).T
            for i, t in enumerate(self.times):
                rankings.insert(i*2 + 1, f"[SE_{t}]", error_matrix[:, i])
        
        if sort_by == "Team":
            rankings = rankings.sort_index(ascending=True)
        elif sort_by == "Average":
            rankings = rankings.loc[avgs.sort_values(ascending=False).index]
        elif sort_by == "Time":
            if sort_time is None: sort_time = self.times[0]     # Default sort time is first block
            rankings = rankings.sort_values(by=sort_time, ascending=False)
        else:   # Invalid sort type
            raise ValueError("sort_by must be one of: 'Team', 'Time', 'Average'")

        return rankings #.reset_index(names="Team")
    
    def get_hgas(self, include_errors: bool = False, as_str: bool = False):
        """Return the home-ground parameter(s) of a model, such as in a Series or DataFrame."""
        raise NotImplementedError()
    
    def summary(
            self, verbose: int = 2, include_errors: bool = True,
            print_summary: bool = True, sort_by: str = "Average", wrap_len: int = 100
        ) -> str:
        """Print a string for a nicely formatted summary of the model.

        Args:
            verbose (int, optional): Verbosity level. Defaults to 2.
                - 0: only prints model metadata.
                - 1: adds average strengths and HGA parameters.
                - 2: adds all strength parameters.
            include_errors (bool, optional): If True, adds errors to strength output. Defaults to True.
            print_summary (bool, optional): If True, prints the summary. Defaults to True.
            sort_by (str, optional): See `get_ranking()`. Defaults to "Average".
            wrap_len (int, optional): Char length to wrap text. Defaults to 100.

        Returns:
            str: Model summary.
        """
        # TODO print n_levels or some other HIE consideration
        self._check_fitted()
        stats = model_statistics(self)
        padding_bar = "=" * wrap_len
        padding_bar_short = "-" * 25

        s = f"{padding_bar}\nMODEL SUMMARY: {self.__class__.__name__}\n{padding_bar}\n\n"
        
        # Fit Statistics
        stat_pad = 10
        s += f"{'Model Statistics'}\n"
        s += f"{padding_bar_short}\n"
        s += f"{'# Teams:':<{stat_pad}} {self.n_teams:<15} {'# Obs:':<{stat_pad}} {int(stats['n_obs'])}\n"
        s += f"{'# Times:':<{stat_pad}} {self.n_times:<15} {'# Params:':<{stat_pad}} {self.n_params_active}\n"
        s += f"{'LLH:':<{stat_pad}} {stats['llh']:.3f}\n"
        s += f"{'AIC:':<{stat_pad}} {stats['aic']:.3f}\n"
        s += f"{'BIC:':<{stat_pad}} {stats['bic']:.3f}\n"
        s += f"{"# Iter:":<{stat_pad}} {stats['n_iter']}\n"
        s += f"\n{"Constraint team:"} {self.teams[self._constraint_team_idx]}\n"

        # Parameters
        if verbose:
            s += f"\n{'Team Rankings'}\n"
            s += f"{padding_bar_short}\n"
            rankings = self.get_ranking(sort_by=sort_by, include_errors=include_errors).round(3)
            if verbose == 1: 
                rankings = rankings["Average"]
                s += rankings.to_string() + "\n"
            else:
                longest_col_name = max([len(str(col)) for col in rankings.columns])
                s += rankings.to_string(col_space=longest_col_name, line_width=wrap_len) + "\n"

            hga_str = self.get_hgas(as_str=True, include_errors=include_errors)
            if hga_str:
                s += f"\n{'Home-Ground Advantage'}\n"
                s += f"{padding_bar_short}\n"
                s += hga_str + "\n"

        s += "\n" + padding_bar
        if print_summary: print(s)
        return s
    
    ## ===== ODDS AND PROBS CALCULATIONS ===== ##
    
    @staticmethod
    def _calculate_odds(it: float, jt: float, h: float) -> float:
        """Calculate log-odds for `{i beats j | t, venue}`."""
        return it - jt + h
    
    @staticmethod
    def _calculate_prob(it: float, jt: float, h: float) -> float:
        """Calculate probability for `{i beats j | t, venue}`."""
        return expit(BaseBradleyTerry._calculate_odds(it, jt, h))
    
    def _get_venue_map(self, i, j) -> dict:
        raise NotImplementedError()

    def get_odds(self, i, j, t, venue: Literal["home", "away", "neutral"] = "neutral") -> float:
        theta_it = self.get_param(team=i, time=t)
        theta_jt = self.get_param(team=j, time=t)
        hga = self._get_venue_map(i, j)[venue]
        return self._calculate_odds(theta_it, theta_jt, hga)

    def get_prob(self, i, j, t, venue: Literal["home", "away", "neutral"] = "neutral") -> float:
        return expit(self.get_odds(i, j, t, venue))
    
    ## ===== EXTRA CLASS FNS ===== ##

    def __str__(self):
        if self.params is None:
            return f"Unfitted {self.__class__.__name__} model."
        return f"Fitted {self.__class__.__name__} model with {self.n_teams} teams."

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class BaseHierarchicalBT(BaseBradleyTerry):
    """Base class for Hierarchical Bradley-Terry models. 
    
    Adds params for the relationship matrix and relevant getters to `BaseBradleyTerry` class.
    """
    def __init__(self):
        super().__init__()
        self._rel_mat = None
        self._mask = None
        self.levels = None
        self.n_levels = 0

    def _set_data(self, data: dict, rel_mat: DataFrame):
        self._rel_mat = rel_mat
        super()._set_data(data)

    def _validate_rel_mat(self):
        """Validate the relationship matrix, which must have been set already to `self.rel_mat`"""
        # Check square matrix
        if self._rel_mat.shape != (self.n_teams, self.n_teams):
            raise ValueError(f"rel_mat must be {self.n_teams}x{self.n_teams}, got {self._rel_mat.shape}")
    
    def _finalise_params(self):
        """Finishes extracting hierarchical model-specific params after general params are set."""
        # Validate the relationship matrix AFTER the other params have been set
        self._validate_rel_mat()
        self._mask = ~np.eye(self.n_teams, dtype=bool)
        self.levels = np.unique(self._rel_mat.values[self._mask]).tolist()
        self.n_levels = len(self.levels)
        super()._finalise_params()

    def fit(self, data: DataFrame, rel_mat: DataFrame, verbose: bool = False):
        """Fit the Bradley-Terry model to the data. The model parameters are estimated by minimising
        the negative log-likelihood. Currently uses the BFGS method, which requires the gradient of
        the objective function, i.e., the (negative) score statistic.

        See `_init_params()` for setting initial parameter vector.

        Args:
            data (DataFrame): Data for the model.
            rel_mat (DataFrame): Relationship matrix for the teams.
            verbose (bool, optional): If true, prints convergence messages from `scipy.minimize()`.
                Defaults to False.

        Returns:
            self: Fitted model.
        """
        self._set_data(data, rel_mat)
        return self._fit_model(verbose=verbose)
    
    def get_levels(self):
        return self.levels
    
    def get_n_levels(self):
        return self.n_levels
    
    def _get_rel_mat(self) -> ndarray:
        return self._rel_mat.values
    
    def get_relationship(self, i, j):
        i_idx = self.teams.index(i)
        j_idx = self.teams.index(j)
        return self._rel_mat[i_idx][j_idx]

    def _get_hga_param(self, team = None, level = None) -> float:
        raise NotImplementedError()
