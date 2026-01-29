import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize

# TODO make abstract base class
# TODO consider staticmethods (LLH, odds, prob)
# TODO naming for odds (when it's actually log-odds)
# TODO type hint fields? eg self.data: dict, self.teams: list
# TODO enum for things like venue types
# TODO consider making a wrapper for static models
class BaseBradleyTerry():
    """Abstract base class for our Bradley-Terry models. Mainly specifies the methods that each 
    model will need to implement, primarily the log-likelihood function.
    """
    venue_keys = {"home", "away", "neutral"}

    def __init__(self):
        """Initialise a Bradley-Terry model. Simply sets all fields to None. Setting should be done
        before model fitting with `_set_data()` method.
        """
        # Dataset characteristics
        self.data = None
        self.n_obs = None
        self.teams = None
        self.n_teams = None
        self.times = None
        self.n_times = None

        # Model fit items
        self.constraint_team_idx = 0
        self.params = None
        self.n_params = None
        self._fit_summary = None
        self._hess_inv = None
        self.errors = None

    def _set_data(self, data: dict):
        """Validate and set data-related parameters.
        Expected data structure is a dictionary with time block names as keys and win matrices as values.
        Currently expecting the win matrices to be pandas DataFrames with team names as column names.
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
                
    def _finalise_params(self):
        """Default setting number of parameters."""
        self.n_params = self.get_n_params()

    def _init_params(self) -> ndarray:
        """Initialise model parameters for fitting. Currently just using a zero vector for all
        initial parameters.

        Returns:
            ndarray: Initial parameter vector.
        """
        return np.zeros(self.n_params)
    
    def _check_fitted(self) -> bool:
        """Check if the model parameters have been fit, and raises an exception if not.

        Returns:
            bool: True if the model parameters have been fit.
        """
        if self.params is None:
            raise RuntimeError("Model parameters are not yet fitted.")

        return True

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
        initial_theta = self._init_params()

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
    
    def rebase_abilities(self, method: str = "first_index", custom_team: str = None):
        """Helper function to rebase the estimated team strength parameters such that a chosen team 
        has zero strength in each time block. Needed to enforce model identifiability constraint(s).

        Args:
            method (str): Method for rebasing, options are
                * "first_index" which takes the team with index 0 (default);
                * "worst" which takes the team with worst strength on average; or,
                * "custom" which lets the user specify any valid team name.
            custom_team (str): If `method="custom"`, specify which team to set to zero.
        """
        self._check_fitted()

        if method == "worst":
            rankings = self.get_ranking(sort_by="Average")
            self.constraint_team_idx = np.argmin(rankings["Average"])
        if method == "custom" and custom_team is not None:
            # TODO try-except for team name?
            self.constraint_team_idx = self.teams.index(custom_team)
        else: # Assume default
            self.constraint_team_idx = 0

        for t in range(self.n_times):
            start_index = t * self.n_teams
            end_index = start_index + self.n_teams
            self.params[start_index:end_index] -= self.params[start_index + self.constraint_team_idx]

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

        # TODO might need to overwrite for static class but worry about this later
        # TODO add hga params that don't get estimated (e.g. in TSI models) to remove list
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
    
    def _get_strength_idx(self, i, t) -> int:
        """Get the parameter vector index of a team strength.

        Args:
            i: Team index.
            t: Time index.
        """
        return i + self.n_teams * t
    
    def _get_hga_idx(self) -> int:
        raise NotImplementedError()

    def get_param(self) -> float:
        # Function for user to get a specific parameter from the model, since directly working with
        # the vectorised array is not convenient
        raise NotImplementedError()
    
    def _get_params(self) -> ndarray:
        """Get the parameter vector for the model. Recommend using `get_param()` method instead.
        See class docstring for format, since we need to vectorise for use in `fit()` method."""
        return self.params
    
    def get_n_params(self) -> int:
        """Get the number of parameters in the model."""
        raise NotImplementedError()
    
    def get_ranking(self, sort_by: str = None) -> DataFrame:
        """
        Gets the estimated team abilities for all years, with customizable sorting.

        Args:
            sort_by (str): Specifies the column to sort the teams by. Options include:
                        - 'Year X' (e.g., 'Year 1', 'Year 2') to sort by a specific year's ability.
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
        results_df["Average"] = results_df.mean(axis=1)
        
        # Various options for sorting while I think about a good standard approach
        # TODO consider alternatives, like rebasing relative to average strength, or displaying 
        # probability of winning against an average team (strength 0)
        if sort_by == "Team":
            results_df = results_df.sort_index(ascending=True)
        elif sort_by in results_df.columns or sort_by == "Average":
            results_df = results_df.sort_values(by=sort_by, ascending=False)
        else:
            # Default sort_by is first time block
            results_df = results_df.sort_values(by=results_df.columns[0], ascending=False)

        results_df = results_df.reset_index(names="Team")

        return results_df
    
    def summary(self) -> str:
        """Return a string for a pretty printed summary of the model."""
        raise NotImplementedError()
    
    @staticmethod
    def _calculate_odds(i: float, j: float, **kwargs) -> float:
        """Calculate log-odds for `{i beats j}`. Static method for each subclass."""
        raise NotImplementedError()

    def get_odds(self, i: str | int, j: str | int, **kwargs) -> float:
        raise NotImplementedError()
    
    @staticmethod
    def _calculate_prob(i: float, j: float, **kwargs) -> float:
        """Calculate probability for `{i beats j}`. Static method for each subclass."""
        raise NotImplementedError()

    def get_prob(self, i: str | int, j: str | int, **kwargs) -> float:
        # TODO return expit(get_odds(params)) ?
        raise NotImplementedError()

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
        self.n_levels = None

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

    def fit(self, data: DataFrame, rel_mat, verbose: bool = False):
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
        initial_theta = self._init_params()

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
