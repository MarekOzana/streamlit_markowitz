"""
Unit tests on HIstorical_Risk_Returns streamlit module
(c) 2024-10 Marek Ozana
"""

from datetime import date
import scripts.Historical_Risk_Return as rr
import polars as pl
from polars.testing import assert_frame_equal
import pytest


@pytest.fixture
def r_m():
    tickers = ["SEB Hybrid G-SEK", "Sweden Eq"]
    start_dt = date(2023, 3, 29)
    r_m = (
        rr.get_monthly_rets(tickers=tickers, start_dt=start_dt)
        .filter(pl.col("date") < date(2024, 10, 1))
        .with_columns(pl.col("r").round(4))
    )
    return r_m


def test_monthly_rets(r_m: pl.DataFrame):
    """Unit test on get_monthly_rets function"""
    assert r_m.shape == (38, 3)

    exp_r_m = pl.DataFrame(
        {
            "date": [
                date(2023, 3, 31),
                date(2023, 4, 28),
                date(2023, 5, 31),
                date(2023, 6, 30),
                date(2023, 7, 31),
                date(2023, 8, 31),
                date(2023, 9, 29),
                date(2023, 10, 31),
                date(2023, 11, 30),
                date(2023, 12, 29),
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 3, 28),
                date(2024, 4, 30),
                date(2024, 5, 31),
                date(2024, 6, 28),
                date(2024, 7, 31),
                date(2024, 8, 30),
                date(2024, 9, 30),
                date(2023, 3, 31),
                date(2023, 4, 28),
                date(2023, 5, 31),
                date(2023, 6, 30),
                date(2023, 7, 31),
                date(2023, 8, 31),
                date(2023, 9, 29),
                date(2023, 10, 31),
                date(2023, 11, 30),
                date(2023, 12, 29),
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 3, 28),
                date(2024, 4, 30),
                date(2024, 5, 31),
                date(2024, 6, 28),
                date(2024, 7, 31),
                date(2024, 8, 30),
                date(2024, 9, 30),
            ],
            "security": [
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "SEB Hybrid G-SEK",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
                "Sweden Eq",
            ],
            "r": [
                0.0018,
                -0.006,
                0.0076,
                0.0053,
                0.0243,
                -0.0071,
                -0.0023,
                0.0014,
                0.0356,
                0.0315,
                0.0099,
                0.0019,
                0.0174,
                -0.005,
                0.0151,
                0.0009,
                0.0175,
                0.0135,
                0.0135,
                0.039,
                0.0303,
                -0.0271,
                0.0215,
                -0.0033,
                -0.0374,
                -0.0201,
                -0.0352,
                0.0915,
                0.0818,
                -0.0168,
                0.0384,
                0.0565,
                0.0026,
                0.0383,
                -0.0153,
                0.0283,
                -0.0003,
                0.0134,
            ],
        }
    )
    assert_frame_equal(r_m, exp_r_m)


def test_calc_rets(r_m):
    """Unit test on calc_rets function"""
    rets = rr.calc_rets(r_m, scale=12).sort(by=["security"])
    exp_rets = pl.DataFrame(
        {"security": ["SEB Hybrid G-SEK", "Sweden Eq"], "r": [0.116622522, 0.187356642]}
    )
    assert_frame_equal(rets, exp_rets)


def test_calc_vol(r_m):
    """Unit test on calc_vol function"""
    vol = rr.calc_vols(r_m, scale=12).sort(by=["security"])
    exp_vol = pl.DataFrame(
        {
            "security": ["SEB Hybrid G-SEK", "Sweden Eq"],
            "vol": [0.04255686, 0.129308949],
        }
    )
    assert_frame_equal(vol, exp_vol)


def test_calc_portf_rets():
    # Sample monthly returns data
    r_m = pl.DataFrame(
        {
            "date": [date(2024, 9, 1), date(2024, 9, 1), date(2024, 9, 1)],
            "dt_end": [date(2024, 9, 30), date(2024, 9, 30), date(2024, 9, 30)],
            "security": ["AT1 EUR", "SEB Obl", "SEB Value"],
            "r": [0.01477, 0.0069, -0.00626],
        }
    )

    # Sample security to weight mapping
    sec2w = {
        "AT1 EUR": 0.4,
        "SEB Obl": 0.3,
        "SEB Value": 0.3,
    }

    # Call the function with the sample data
    rets = rr.calc_portf_rets(r_m, sec2w)

    # Check if the result is a DataFrame
    assert isinstance(rets, pl.DataFrame)

    # Check if the result has the expected columns
    expected_columns = ["date", "security", "r"]
    assert all(col in rets.columns for col in expected_columns)

    # Check if the result has the correct data types
    assert rets["date"].dtype == pl.Date
    assert rets["security"].dtype == pl.String
    assert rets["r"].dtype == pl.Float64

    # Check if the result has the correct values
    exp_rets = pl.DataFrame(
        {
            "date": [date(2024, 9, 1)],
            "security": ["PORTF"],
            "r": [0.0060999999999999995],
        }
    )
    assert_frame_equal(rets, exp_rets)


def test_calc_portfolio_metrics():
    """Unit test on calc_portfolio_metrics function"""
    tickers = ["SEB Hybrid G-SEK", "Sweden Eq", "Sweden Govt"]
    r_m = (
        rr.get_monthly_rets(tickers=tickers, start_dt=date(2023, 3, 29))
        .filter(pl.col("date") < date(2024, 10, 1))
        .with_columns(pl.col("r").round(4))
    )
    portfolios = rr.calc_portfolio_metrics(r_m, tickers)

    # Check if the result is a DataFrame
    assert isinstance(portfolios, pl.DataFrame)

    # Check if the result has the expected columns
    expected_columns = ["name", "r", "vol", "r2vol", "w0", "w1", "w2"]
    assert all(col in portfolios.columns for col in expected_columns)

    # Check if the result has the correct data types
    assert portfolios["name"].dtype == pl.String
    assert portfolios["r"].dtype == pl.Float64
    assert portfolios["vol"].dtype == pl.Float64
    assert portfolios["r2vol"].dtype == pl.Float64
    assert portfolios["w0"].dtype == pl.Float64
    assert portfolios["w1"].dtype == pl.Float64
    assert portfolios["w2"].dtype == pl.Float64

    # Check if the result has the correct number of rows
    assert portfolios.shape == (66, 7)

    # Check if the result has the correct values for a specific portfolio
    specific_portfolio = portfolios.filter(pl.col("name") == "50%/30%/20%")
    assert specific_portfolio["r"].item() == pytest.approx(0.1226)
    assert specific_portfolio["vol"].item() == pytest.approx(0.0611)
    assert specific_portfolio["r2vol"].item() == pytest.approx(2.0054)
    assert specific_portfolio["w0"].item() == pytest.approx(0.5)
    assert specific_portfolio["w1"].item() == pytest.approx(0.3)
    assert specific_portfolio["w2"].item() == pytest.approx(0.2)
