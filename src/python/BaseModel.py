import numpy as np
from pandas import DataFrame
from scipy.optimize import minimize

# TODO make abstract base class
class BaseBradleyTerry():
    def __init__(self, data: DataFrame):
        # TODO write init overloads for non-pandas use cases
        # TODO change asserts to nicer input validation
        # TODO look at fields and methods provided by scikit learn models for inspiration
        # TODO refactor data input to be with `fit` method, not init
        assert type(data) == DataFrame and data.shape[0] == data.shape[1]

        self.data = data
        self.teams = data.columns
        self.nteams = len(data.columns)
        self.params = None
        self.nparams = None
        self.fit_summary = None # TODO is this needed?

    def _init_params(self):
        raise NotImplementedError()

    def _log_likelihood(self, params: list) -> float:
        raise NotImplementedError()

    def _score(self, params: list) -> list:
        raise NotImplementedError()

    def _hessian(self, params: list) -> list:
        raise NotImplementedError()

    def fit(self, verbose: bool = False):
        initial_theta = self._init_params()
        result = minimize(
            fun = self._log_likelihood,
            x0 = initial_theta,
            method = "BFGS",
            jac = self._score,
            # options = # TODO check options, could be a param itself?
        )

        if result.success:
            self.params = result.x
            self._rebase_abilities()
            self.fit_summary = result
            if verbose:
                # TODO improve verbose output(s) / levels
                print("Success")
        else:
            print(f"Error encountered during fit: {result.message}")

        return result
    
    def _rebase_abilities(self):
        if self.params is None:
            print("Error: model hasn't been fitted.")
            return
        
        self.params = self.params - np.min(self.params)

    def get_ranking(self):
        assert self.params is not None

        results = {
            "Team": self.teams,
            "Ability": self.params
        }

        return DataFrame(results).sort_values(by="Ability", ascending=False)

    def get_odds(self):
        pass

    def get_prob(self):
        pass

    # TODO add more get/set methods
    # TODO check scikit learn model impls for inspo on other convenient functions