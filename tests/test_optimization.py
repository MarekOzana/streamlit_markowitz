"""
Unit tests on Optimization
"""

import numpy as np
from src.optimization import find_min_var_portfolio


def test_correct_output_with_known_input():
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
