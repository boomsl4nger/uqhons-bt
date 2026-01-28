# TODO make nestedness checker for time axis and combinations of such

from numpy import log
from scipy.stats import chi2

from BaseModel import BaseBradleyTerry

def calculate_aic(nll: float, k: int) -> float:
    """Calculate the Akaike information criterion (AIC) for a model.

    See: https://en.wikipedia.org/wiki/Akaike_information_criterion

    Args:
        nll (float): Negative log-likelihood.
        k (int): Number of estimated parameters.

    Returns:
        float: AIC value.
    """
    return 2 * k + 2 * nll

def calculate_bic(nll: float, k: int, n: int) -> float:
    """Calculate the Bayesian information criterion (BIC) for a model.

    See: https://en.wikipedia.org/wiki/Bayesian_information_criterion

    Args:
        nll (float): Negative log-likelihood.
        k (int): Number of estimated parameters.
        n (int): Number of observations (sample size).

    Returns:
        float: BIC value.
    """
    return k * log(n) + 2 * nll

def compare_deviances(m1: BaseBradleyTerry, m2: BaseBradleyTerry) -> dict:
    """Perform likelihood ratio test (LRT) by comparison of deviance for two models. 
    Assumes that M1 is nested in M2.

    Args:
        m1 (BaseBradleyTerry): Nested model.
        m2 (BaseBradleyTerry): Alternative model.

    Returns:
        dict: Results containing `"stat", "df", "p_value"`.
    """
    D = 2 * (m1._fit_summary.fun - m2._fit_summary.fun)
    df = m2.n_params - m1.n_params

    if df <= 0:
        raise ValueError("Model M1 must have less parameters than M2.")
    
    p_val = chi2.sf(D, df)

    return {
        "stat": D,
        "df": df,
        "p_value": p_val
    }

def is_nested(m1_name: str, m2_name: str) -> bool:
    """Check if model M1 is nested in model M2 based on our Bradley-Terry model relationships.
    The order effects nestedness hierarchy is:
    `VANBT > CHABT > (CHIBT OR TSABT) > TSIBT`

    Args:
        m1_name (str): Name of null model.
        m2_name (str): Name of alternative model.

    Returns:
        bool: True if M1 is nested in M2, otherwise False.
    """
    ranks = {
        "DyVANBT": 0,
        "DyCHABT": 1,
        "DyCHIBT": 2,
        "DyTSABT": 2,
        "DyTSIBT": 3
    }
    
    if m1_name not in ranks or m2_name not in ranks:
        raise ValueError(f"Unknown model types: {m1_name}, {m2_name}")

    return ranks[m1_name] < ranks[m2_name]

def simple_report(m1: BaseBradleyTerry, m2: BaseBradleyTerry, n: int):
    """Run a basic test suite for two given BT models.

    Args:
        m1 (BaseBradleyTerry): Null model.
        m2 (BaseBradleyTerry): Alternative model.
        n (int): Number of observations. TODO remove and use `m1.n_obs` (check same as in M2).

    Returns:
        dict: Results containing `"m1", "m2", "lrt"`. Each model contains `"name", "aic", "bic"`.
    """
    m1_nll = m1._fit_summary.fun
    m2_nll = m2._fit_summary.fun
    m1_k = m1.n_params
    m2_k = m2.n_params
    
    results = {
        "m1": {
            "name": m1.__class__.__name__,
            "aic": calculate_aic(m1_nll, m1_k), 
            "bic": calculate_bic(m1_nll, m1_k, n)
        },
        "m2": {
            "name": m2.__class__.__name__,
            "aic": calculate_aic(m2_nll, m2_k), 
            "bic": calculate_bic(m2_nll, m2_k, n)
        },
    }
    
    if is_nested(m1.__class__.__name__, m2.__class__.__name__):
        results["lrt"] = compare_deviances(m1, m2)
    else:
        results["lrt"] = "Incompatible"
    
    return results
