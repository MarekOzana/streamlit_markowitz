"""
Unit tests on Optimization
"""

import numpy as np
from src.optimization import find_min_var_portfolio
from src.optimization import calc_eff_front


def test_find_min_var_portfolio():
    # Define a simple scenario with known inputs and outputs.
    exp_rets = np.array([0.1, 0.2])
    cov = np.array([[0.01, 0], [0, 0.04]])
    r_min = 0.15
    w_max = 1
    expected_weights = np.array([0.5, 0.5])
    expected_return = 0.15
    expected_volatility = 0.11180339

    weights, r_opt, vol_opt = find_min_var_portfolio(exp_rets, cov, r_min, w_max)

    # Verify the output against the expected values
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_almost_equal(r_opt, expected_return, decimal=5)
    np.testing.assert_almost_equal(vol_opt, expected_volatility, decimal=3)


def test_calc_eff_front():
    exp_rets = np.array([0.078, 0.04, 0.044, 0.064, 0.1, 0.068, 0.08])
    cov = (
        np.array(
            [
                [0.13, -0.0, 0.04, 0.08, 0.21, 0.05, 0.03],
                [-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0],
                [0.04, -0.0, 0.18, 0.07, 0.08, 0.06, -0.1],
                [0.08, -0.0, 0.07, 0.18, 0.24, 0.07, -0.13],
                [0.21, 0.0, 0.08, 0.24, 1.19, 0.11, 0.14],
                [0.05, -0.0, 0.06, 0.07, 0.11, 0.08, -0.03],
                [0.03, 0.0, -0.1, -0.13, 0.14, -0.03, 1.23],
            ]
        )
        / 100
    )
    frnt = calc_eff_front(exp_rets, cov)
    assert set(frnt.keys()) == {"vols", "rets"}
    np.testing.assert_almost_equal(
        frnt["rets"],
        [0.04, 0.047, 0.055, 0.063, 0.07, 0.078, 0.085, 0.093, 0.1],
        decimal=3,
    )
    np.testing.assert_almost_equal(
        frnt["vols"],
        [0.0, 0.006, 0.012, 0.018, 0.024, 0.033, 0.051, 0.078, 0.109],
        decimal=3,
    )
