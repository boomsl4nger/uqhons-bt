import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize

# TODO make abstract base class
class BaseBradleyTerry():
    def __init__(self):
        self.data = None
        self.n_teams = None
        self.teams = None
        self.n_params = None
        self.params = None
        self._fit_summary = None

    def _log_likelihood(self, params: ndarray) -> float:
        raise NotImplementedError()

    def _score(self, params: ndarray) -> ndarray:
        raise NotImplementedError()

    def _hessian(self, params: ndarray) -> ndarray:
        raise NotImplementedError()
    
    def _set_data(self, data: DataFrame):
        raise NotImplementedError()

    def _init_params(self) -> ndarray:
        return np.zeros(self.n_params)

    def fit(self, data: DataFrame, verbose: bool = False):
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
        if self.params is None:
            print("Error: model hasn't been fitted.")
            return
        
        I = self.n_teams
        self.params[1:I] = self.params[1:I] - np.min(self.params[1:I])
    
    def get_teams(self):
        return self.teams
    
    def get_n_teams(self):
        return self.n_teams
    
    def get_params(self, tidy_str: bool = False):
        # TODO handle tidy printing in overrides since params is a single vectorised array of 
        # the different params, like [1:I+1] is team strengths, rest is HFA's (varies per model)
        return self.params
    
    def get_n_params(self) -> int:
        raise NotImplementedError()

    def get_ranking(self):
        raise NotImplementedError()

    def get_odds(self, i: int, j: int, **kwargs) -> float:
        # TODO check kwargs is the right way to generalise for HFA
        raise NotImplementedError()

    def get_prob(self, i: int, j: int, **kwargs) -> float:
        # TODO return expit(get_odds(params)) ?
        raise NotImplementedError()

    # def __str__(self):
    #     pass

    # def __repr__(self):
    #     pass

    # TODO check scikit learn model, BT2, choix impls for inspo on other convenient functions
    # TODO check other good OOP practices for methods like __str__