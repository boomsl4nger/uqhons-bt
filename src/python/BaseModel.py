import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize

# TODO make abstract base class
class BaseBradleyTerry():
    """Abstract base class for Bradley-Terry models. Mainly specifies the methods that each specific
    model will need to implement.
    """

    def __init__(self):
        """Initialise a Bradley-Terry model. Simply sets all fields to None. Setting should be done
        before model fitting with `_set_data()` method.
        """
        self.data = None
        self.n_teams = None
        self.teams = None
        self.n_params = None
        self.params = None
        self._fit_summary = None

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
    
    def _set_data(self, data: DataFrame):
        """Set class variables based on the given data. This is done just before model fitting
        rather than on init, following scikit-learn syntax.

        Args:
            data (DataFrame): Data for the model.
        """
        raise NotImplementedError()

    def _init_params(self) -> ndarray:
        """Initialise model parameters for fitting. Currently just using a zero vector for all
        initial parameters.

        Returns:
            ndarray: Initial parameter vector.
        """
        return np.zeros(self.n_params)

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
                print("Successfully fit. Summary below:\n")
                print(self._fit_summary)
        else:
            print(f"Error: {result.message}")

        return self
    
    def _rebase_abilities(self):
        """Helper function to rebase the estimated team strength parameters such that the 'worst'
        team has zero strength. Needed to enforce model identifiability constraint.

        TODO: reference?
        """
        if self.params is None:
            print("Error: model hasn't been fitted.")
            return
        
        I = self.n_teams
        self.params[0:I] -= np.min(self.params[0:I])
    
    def get_teams(self) -> list:
        """Get the list of team names for the model."""
        return self.teams
    
    def get_n_teams(self) -> int:
        """Get the number of teams in the model."""
        return self.n_teams
    
    def get_params(self):
        """Get the parameter vector for the model. See class docstring for format, since we need to
        vectorise for use in `fit()` method."""
        return self.params
    
    def get_n_params(self) -> int:
        raise NotImplementedError()
    
    def summary(self) -> str:
        """Return a string for a pretty printed summary of the model."""
        # TODO handle tidy printing in overrides since params is a single vectorised array of 
        # the different params, like [1:I+1] is team strengths, rest is HFA's (varies per model)
        raise NotImplementedError()

    def get_ranking(self):
        """Get the ranking of teams based on their estimated strengths, in descending order. Note
        that the 'worst' team will have zero strength for identifiability.

        Returns:
            DataFrame: Sorted tuples of team name and estimated strength.
        """
        assert self.params is not None

        df = DataFrame(self.params[0:self.n_teams], index=self.teams, columns=["Ability"])

        return df.sort_values(by="Ability", ascending=False).reset_index(names="Teams")

    def get_odds(self, i: int, j: int, **kwargs) -> float:
        # TODO check kwargs is the right way to generalise for HFA
        raise NotImplementedError()

    def get_prob(self, i: int, j: int, **kwargs) -> float:
        # TODO return expit(get_odds(params)) ?
        raise NotImplementedError()

    def __str__(self):
        if self.params is None:
            return f"Unfitted {self.__class__.__name__} model."
        return f"Fitted {self.__class__.__name__} model with {self.n_teams} teams."

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    # TODO add check_fitted method to generically handle unwanted calls before fitting
    # TODO check scikit learn model, BT2, choix impls for inspo on other convenient functions
    # TODO check other good OOP practices for methods like __str__

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

        # Rebase such that the worst team in each year has strength zero
        for t in range(self.n_times):
            # Get indices assuming teams are grouped by year
            # TODO refactor for generic indexing fn
            start_index = t * self.n_teams
            end_index = (t + 1) * self.n_teams

            self.params[start_index:end_index] -= np.min(self.params[start_index:end_index])
    
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
        assert self.params is not None

        # Reshape to (I x T) matrix format
        # TODO account for HGA params in non-van models
        ability_matrix = self.params.reshape(self.n_times, self.n_teams).T

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