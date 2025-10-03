import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize

# TODO make abstract base class
# TODO consider staticmethods (LLH, odds, prob)
# TODO naming for odds (when it's actually log-odds)
class BaseBradleyTerry():
    """Abstract base class for Bradley-Terry models. Mainly specifies the methods that each specific
    model will need to implement.
    """

    def __init__(self):
        """Initialise a Bradley-Terry model. Simply sets all fields to None. Setting should be done
        before model fitting with `_set_data()` method.
        """
        self.data = None
        self.teams = None
        self.n_teams = None
        self.params = None
        self.n_params = None
        self._fit_summary = None
        self._hess_inv = None
        self.errors = None

    def _set_data(self, data):
        raise NotImplementedError()

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
            print(f"Error: model parameters are not fitted.")
            raise # TODO better way to do this?

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
            self._rebase_abilities()
            self._fit_summary = result
            if verbose:
                print("Successfully fit model parameters.\n")
                print(self._fit_summary)
        else:
            print(result.message)

        return self
    
    def _rebase_abilities(self):
        """Helper function to rebase the estimated team strength parameters such that the 'worst'
        team has zero strength. Needed to enforce model identifiability constraint. 
        
        For the static Bradley-Terry models, we assume that the team strength parameters are the 
        first `I` elements in the parameter array, where `I` is the number of teams.
        """
        self._check_fitted()
        self.params[0:self.n_teams] -= np.min(self.params[0:self.n_teams])

    def _calculate_errors(self):
        """Calculate the standard errors for the model parameters using the inverse of the Hessian.
        Hessian should be implemented analytically, but numerical one is available as an element of
        `self._fit_summary`."""
        self._check_fitted()
        
        # We need to consider the zeroed parameter from the identifiability constraint
        # Calculate the full Hessian, then remove row/col and invert (should now be non-singular)
        # Standard errors are the sqrts of diagonal elements (obs Fisher approximates covariance mat)
        # Finally, just set the zeroed parameter SE to zero itself
        zero_idx = np.argmin(self.params)
        hess = self._hessian(self.params)
        hess_reduced = np.delete(np.delete(hess, zero_idx, axis=0), zero_idx, axis=1)

        # TODO wrap in try-except for np.linalg.LinAlgError if still singular...
        self._hess_inv = np.linalg.inv(hess_reduced)
        temp_errors = np.sqrt(np.diag(self._hess_inv))
        self.errors = np.insert(temp_errors, zero_idx, 0)
    
    def get_teams(self) -> list:
        """Get the list of team names in the data."""
        return self.teams
    
    def get_n_teams(self) -> int:
        """Get the number of teams in the data."""
        return self.n_teams
    
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
    
    def get_ranking(self):
        """Get the ranking of teams based on their estimated strengths, in descending order. Note
        that the 'worst' team will have zero strength for identifiability.

        Returns:
            DataFrame: Sorted tuples of team name and estimated strength.
        """
        self._check_fitted()
        dummy = {"Ability": self.params[0:self.n_teams]}
        if self.errors is not None:
            dummy["Error"] = self.errors[0:self.n_teams]

        df = DataFrame(dummy, index=self.teams)
        return df.sort_values(by="Ability", ascending=False).reset_index(names="Teams")

    def __str__(self):
        if self.params is None:
            return f"Unfitted {self.__class__.__name__} model."
        return f"Fitted {self.__class__.__name__} model with {self.n_teams} teams."

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class BaseDyBT(BaseBradleyTerry):
    """Abstract base class for a Dynamic Bradley-Terry model. This is specifically for our discrete
    dynamic models, where each team has a different strength for each time block (year) in the data,
    as well as whatever additional order effects are being modelled (e.g. HFA).
    """
    def __init__(self):
        super().__init__()
        self.times = None
        self.n_times = None

    def _rebase_abilities(self):
        if self.params is None:
            print("Error: model hasn't been fitted.")
            return

        # Rebase such that worst team in each year has strength zero
        for t in range(self.n_times):
            # Get indices assuming teams are grouped by year
            # TODO refactor for generic indexing fn
            start_index = t * self.n_teams
            end_index = (t + 1) * self.n_teams

            self.params[start_index:end_index] -= np.min(self.params[start_index:end_index])

    def _get_param_index(self, i, t) -> int:
        return i + self.n_teams * t
    
    def get_times(self):
        """Get the names of the time blocks in the data. Usually should be years (ints)."""
        return self.times
    
    def get_n_times(self):
        """Get the number of time blocks in the data."""
        return self.n_times

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
