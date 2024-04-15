"""
Utilities for Markowitz Optimization

.. author:: Marek Ozana
.. date:: 2024-03
"""

import numpy as np
import scipy.optimize as sco


def find_min_var_portfolio(
    exp_rets: np.array, cov: np.array, r_min: float = 0, w_max: float = 1
):
    """Find portfolio with minimum variance given constraint return
    Solve the following optimization problem
        min: w.T*COV*w
        subjto: w.T * r_ann >= r_min
                sum(w) = 1
                0 <= w[i] <= w_max for every i
    Parameters
    ==========
        exp_rets: annualized expected returns
        cov: covariance matrix
        r_min: minimum portfolio return (constraint)
        w_max: maximum individual weight (constraint)
    Returns
    =======
        (w, r_opt, vol_opt)
        w: portfolio weights
        r_opt: return of optimal portfolio
        vol_opt: volatility of optimal portfolio
    """

    def calc_var(w, cov):
        """Calculate portfolio Variance"""
        return np.dot(w.T, np.dot(cov, w))

    n_assets = len(exp_rets)
    constraints = [
        # sum(w_i) = 1
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        # sum(r_i * w_i >= r_min)
        {"type": "ineq", "fun": lambda x: np.dot(x.T, exp_rets) - r_min},
    ]
    bounds = tuple((0, w_max) for asset in range(n_assets))  # sequence of (min,max)

    opts = sco.minimize(
        # Objective Function
        fun=calc_var,
        # Initial guess
        x0=n_assets * [1.0 / n_assets],
        # Extra Arguments to objective function
        args=(cov,),
        method="SLSQP",
        options={"ftol": 1e-7, "maxiter": 100},
        bounds=bounds,
        constraints=constraints,
    )
    w = opts["x"]
    r_opt = np.dot(w, exp_rets)
    vol_opt = np.sqrt(calc_var(w, cov))
    return w, r_opt, vol_opt


def calc_eff_front(exp_rets: np.array, cov: np.array) -> dict[str, list]:
    """Calculate effective frontier

    Iteratively find optimal portfoli for list of minimum returns

    Parameters
    ----------
        exp_rets: annualized expected returns
        cov: covariance matrix

    Returns
    -------
        frnt: dict("ret":list(float), "vol":list(float))
        Dictionary with points on the efficient frontier
    """
    N_STEPS: int = 9
    frnt: dict[str, list] = {"rets": list(), "vols": list()}
    for r_min in np.linspace(exp_rets.min(), exp_rets.max(), N_STEPS):
        _, ret, vol = find_min_var_portfolio(exp_rets=exp_rets, cov=cov, r_min=r_min)
        frnt["vols"].append(vol)
        frnt["rets"].append(ret)
    return frnt
