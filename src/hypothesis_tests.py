from numpy import log
from scipy.stats import chi2

# TODO need to avoid circular imports
# from BaseModel import BaseBradleyTerry

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

def calculate_deviance_test_stat(nll1: float, nll2: float) -> float:
    """Calculate the deviance test statistic for two models based on their negative log-likelihoods.

    See: https://en.wikipedia.org/wiki/Deviance_(statistics)

    Args:
        nll1 (float): Negative log-likelihood of model 1 (nested model).
        nll2 (float): Negative log-likelihood of model 2 (alternative model).
    
    Returns:
        float: Test statistic.
    """
    return 2 * (nll1 - nll2)

def model_statistics(model) -> dict:
    """Calculate key statistics for a given Bradley-Terry model."""
    nll = model._fit_summary.fun
    return {
        "name": model.__class__.__name__,
        "n_teams": model.n_teams,
        "n_times": model.n_times,
        "n_params": model.n_params_active,
        "n_obs": model.n_obs,
        "n_iter": model._fit_summary.nit,
        "llh": -nll,
        "aic": calculate_aic(nll, model.n_params_active),
        "bic": calculate_bic(nll, model.n_params_active, model.n_obs)
    }

def compare_deviances(m1, m2) -> dict:
    """Perform likelihood ratio test (LRT) by comparison of deviance for two models. 
    Assumes that M1 is nested in M2.

    See: https://en.wikipedia.org/wiki/Wilks'_theorem

    Args:
        m1: Nested model.
        m2: Alternative model.

    Returns:
        dict: Results containing `"stat", "df", "p_value"`.
    """
    D_value = calculate_deviance_test_stat(m1._fit_summary.fun, m2._fit_summary.fun)
    df = m2.n_params_active - m1.n_params_active

    if df <= 0:
        raise ValueError("Model M1 must have less parameters than M2.")
    
    p_val = chi2.sf(D_value, df)

    return {
        "stat": D_value,
        "df": df,
        "p_value": p_val
    }

def is_nested(m1, m2) -> bool:
    """Check if model M1 is nested in model M2 based on our Bradley-Terry model relationships.
    Considers both order effect and temporal complexity.

    The order effects nestedness is:
    `VANBT > CHABT > TSABT`
    Nestedness for the HIEBT models is non-trivial.

    Args:
        m1: Null model.
        m2: Alternative model.

    Returns:
        bool: True if M1 is nested in M2, otherwise False.
    """
    # TODO: add nestedness checking for generic HIE models
    ranks = {"VANBT": 0, "CHABT": 1, "CHIBT": 2, "TSABT": 2, "TSIBT": 3}
    m1_name = m1.__class__.__name__
    m2_name = m2.__class__.__name__
    
    if m1_name not in ranks or m2_name not in ranks:
        raise ValueError(f"Unknown model types: {m1_name}, {m2_name}")

    # A model M1 is nested in M2 if:
    # 1. Temporal: M1 has no more time blocks than M2.
    # 2. Structural: M1 rank <= M2 rank AND they aren't parallel.
    # 3. Identifiability: They aren't the exact same model.
    
    time_nested = m1.n_times <= m2.n_times
    struct_nested = (ranks[m1_name] <= ranks[m2_name]) and not (ranks[m1_name] == ranks[m2_name] and m1_name != m2_name)
    is_not_identical = not (m1_name == m2_name and m1.n_times == m2.n_times)

    return time_nested and struct_nested and is_not_identical

def simple_report(m1, m2, print_report: bool = True, wrap_len: int = 85) -> dict:
    """Run a basic test suite for two given BT models.

    Args:
        m1: Null model.
        m2: Alternative model.

    Returns:
        dict: Results containing `"m1", "m2", "lrt"`. Each model contains `"name", "aic", "bic"`.
    """
    # TODO rename to "anova" or similar
    # TODO allow for multiple models to be passed
    results = {}
    for i, model in enumerate([m1, m2]):
        results[f"m{i+1}"] = {
            "name": model.__class__.__name__,
            "n_times": model.n_times,
            "aic": calculate_aic(model._fit_summary.fun, model.n_params_active), 
            "bic": calculate_bic(model._fit_summary.fun, model.n_params_active, model.n_obs)
        }
    
    nested_bool = is_nested(m1, m2)
    if nested_bool:
        results["lrt"] = compare_deviances(m1, m2)
    else:
        results["lrt"] = "Incompatible"

    if print_report:
        padding = "=" * wrap_len
        s = padding + "\n"
        s += f"Bradley-Terry Model Comparison\n"
        s += "-" * len("Bradley-Terry Model Comparison") + "\n\n"
        
        s += f"{'Model':<10} {'df':<5} {'NLL':<12} {'AIC':<12} {'BIC':<12} {'Deviance':<10} {'p'}\n"
        s += f"{results['m1']['name']:<10} {m1.n_params_active:<5} {m1._fit_summary.fun:<12.2f} {results['m1']['aic']:<12.2f} {results['m1']['bic']:<12.2f}\n"
        m2_line = f"{results['m2']['name']:<10} {m2.n_params_active:<5} {m2._fit_summary.fun:<12.2f} {results['m2']['aic']:<12.2f} {results['m2']['bic']:<12.2f}"
        
        if nested_bool:
            p = results["lrt"]["p_value"]
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            
            if p <= 0.001: sig_marker = "***"
            elif p <= 0.01: sig_marker = "**"
            elif p <= 0.05: sig_marker = "*"
            elif p <= 0.1: sig_marker = "."
            else: sig_marker = ""

            m2_line += f" {results['lrt']['stat']:<10.3f} {p_str} {sig_marker}\n\n"
            m2_line += f"{"Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"}\n"
        else:
            m2_line += f" {'NA':<10} {'NA'}\n"

        s += m2_line
        s += padding

        print(s)
    
    return results
