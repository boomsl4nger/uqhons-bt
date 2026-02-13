import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize
from scipy.special import expit
from typing import Literal

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

        # Model fit items
        self.constraint_team_idx = 0
        self.params = None
        self.n_params = 0
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

        first = data[self.times[0]]

        if self.venue_keys is None: # Vanilla model
            if not isinstance(first, DataFrame):
                raise ValueError(f"{self.__class__.__name__} requires {{time: DataFrame}} structure.")
            ref_df = first
        else:                       # HGA models
            if not (isinstance(first, dict) and "home" in first):
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

        # Data validation loop
        for t in self.times:
            for df in iter_matrices(data[t], t):
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
                
    def _finalise_params(self):
        """Default setting number of parameters."""
        self.n_params = self.get_n_params()
    
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
        # TODO maybe have method param, kwargs for passing options?
        # TODO constrained optimisation for theta_1 = 0 (which gets rebased to theta_(1) = 0 later) 
        # using the L-BFGS-B or similar

        self._set_data(data)
        initial_theta = np.zeros(self.n_params)

        result = minimize(
            fun = self._log_likelihood,
            x0 = initial_theta,
            method = "BFGS",
            jac = self._score,
            options = {"disp": verbose}
        )

        self._fit_summary = result
        if result.success:
            self.params = result.x
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
            self.constraint_team_idx = np.argmin(rankings["Average"])
        elif method == "custom" and custom_team is not None:
            self.constraint_team_idx = self.teams.index(custom_team)
        else: # Assume default
            self.constraint_team_idx = 0

        for t in range(self.n_times):
            start_index = t * self.n_teams
            end_index = start_index + self.n_teams
            self.params[start_index:end_index] -= self.params[start_index + self.constraint_team_idx]

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

        # TODO consider teams parameters in years before the team plays (zero)
        # TODO also hga params that don't get estimated (e.g. in TSI models)
        remove_idx = [self._get_strength_idx(self.constraint_team_idx, t) for t in range(self.n_times)]
        hess = self._hessian(self.params)
        hess_reduced = np.delete(np.delete(hess, remove_idx, axis=0), remove_idx, axis=1)

        try:
            self._hess_inv = np.linalg.inv(hess_reduced)
            self.errors = np.sqrt(np.maximum(np.diag(self._hess_inv), 0)) # Maximum to prevent rare nans
            for idx in sorted(remove_idx):
                self.errors = np.insert(self.errors, idx, 0.0)
        except np.linalg.LinAlgError:
            print("Warning: Hessian is singular.")
    
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
        
    def _get_hga_param(self, team = None, level = None) -> float:
        raise NotImplementedError()
    
    def get_ranking(self, sort_by: str = None, add_average: bool = True) -> DataFrame:
        """
        Gets the estimated team abilities for all years, with customizable sorting.

        Args:
            sort_by (str): Specifies the column to sort the teams by. Options include:
                        - Year (e.g. 1, 2020) to sort by a specific time block. TODO clean this...
                        - 'Average' to sort by the average ability across all years.
                        - 'Team' to sort alphabetically by team name.

        Returns:
            DataFrame: A DataFrame with team names and ability estimates for each year, sorted.
        """
        self._check_fitted()

        # Reshape to (I x T) matrix format
        n_strength_params = self.n_times * self.n_teams
        ability_matrix = self.params[:n_strength_params].reshape(self.n_times, self.n_teams).T

        results_df = DataFrame(
            ability_matrix, 
            index = self.teams, 
            columns = self.times
        )
        
        # Add an average column for potential sorting usability
        if add_average: results_df["Average"] = results_df.mean(axis=1)
        
        if sort_by == "Team":
            results_df = results_df.sort_index(ascending=True)
        elif sort_by in results_df.columns or (sort_by == "Average" and add_average):
            results_df = results_df.sort_values(by=sort_by, ascending=False)
        else:
            # Default sort_by is first time block
            results_df = results_df.sort_values(by=results_df.columns[0], ascending=False)

        return results_df #.reset_index(names="Team")
    
    def summary(self) -> str:
        """Return a string for a pretty printed summary of the model."""
        raise NotImplementedError()
    
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
    """Base class for Hierarchical Bradley-Terry models. Simply adds a `hierarchy` variable and
    relevant getters to `BaseBradleyTerry` class.
    """
    def __init__(self):
        super().__init__()
        self.rel_mat = None
        self.levels = None
        self.n_levels = 0

    def _set_data(self, data: dict, rel_mat: DataFrame):
        self.rel_mat = rel_mat
        super()._set_data(data)

    def _validate_rel_mat(self):
        """Validate the relationship matrix, which must have been set already to `self.rel_mat`"""
        # Check square matrix
        if self.rel_mat.shape != (self.n_teams, self.n_teams):
            raise ValueError(f"rel_mat must be {self.n_teams}×{self.n_teams}, got {self.rel_mat.shape}")
        
        # Check symmetric
        if not (self.rel_mat.values == self.rel_mat.values.T).all():
            raise ValueError("rel_mat must be symmetric.")
    
    def _finalise_params(self):
        """Finishes extracting hierarchical model-specific params after general params are set."""
        # Validate the relationship matrix AFTER the other params have been set
        self._validate_rel_mat()
        self.levels = np.unique(self.rel_mat).tolist()
        self.n_levels = len(self.levels)
        super()._finalise_params()

    def fit(self, data: DataFrame, rel_mat: DataFrame, verbose: bool = False):
        """Fit the Bradley-Terry model to the data. The model parameters are estimated by minimising
        the negative log-likelihood. Currently uses the BFGS method, which requires the gradient of
        the objective function, i.e., the (negative) score statistic.

        See `_init_params()` for setting initial parameter vector.

        Args:
            data (DataFrame): Data for the model.
            TODO
            verbose (bool, optional): If true, prints convergence messages from `scipy.minimize()`.
                Defaults to False.

        Returns:
            self: Fitted model.
        """
        # TODO maybe have method param, kwargs for passing options?
        # TODO constrained optimisation for theta_1 = 0 (which gets rebased to theta_(1) = 0 later) 
        # using the L-BFGS-B or similar

        self._set_data(data, rel_mat)
        initial_theta = np.zeros(self.n_params)

        result = minimize(
            fun = self._log_likelihood,
            x0 = initial_theta,
            method = "BFGS",
            jac = self._score,
            options = {"disp": verbose}
        )

        if result.success:
            self.params = result.x
            self.rebase_abilities()
            self._fit_summary = result
            if verbose:
                print("Successfully fit model parameters.\n")
                print(self._fit_summary)
        else:
            print(f"Model fit error: {result.message}")

        return self

    def _get_hierarchy(self):
        return self.rel_mat
    
    def get_levels(self):
        return self.levels
    
    def get_n_levels(self):
        return self.n_levels
    
    def get_relationship(self, i, j):
        i_idx = self.teams.index(i)
        j_idx = self.teams.index(j)
        return self.rel_mat[i_idx][j_idx]
