"""
Unit tests on data_manager module

.. author:: Marek Ozana
.. date:: 2024-03
"""

import datetime
import pytest
from src.data_manager import DataManager
from pathlib import Path
from numpy.testing import assert_array_almost_equal
import numpy as np
import polars.selectors as cs
import polars as pl


# Fixture to create a DataManager instance
@pytest.fixture
def dm():
    dm = DataManager(
        fund_tbl=Path("tests/data/t_fund.csv"),
        price_tbl=Path("tests/data/t_price.parquet"),
    )
    return dm


def test__init__(dm):
    assert dm.fund_tbl == Path("tests/data/t_fund.csv")
    assert dm.price_tbl == Path("tests/data/t_price.parquet")
    assert dm.t_fund.shape == (7, 5)
    assert dm.t_price.shape == (1881, 3)
    assert dm.corr.shape == (7, 8)
    assert dm.corr["name"].to_list() == [
        "SEB Hybrid",
        "T-Bills",
        "Corps",
        "High Yield",
        "Equity",
        "Climate Focus",
        "Asset Selection",
    ]
    assert dm.ret_vol.shape == (7, 4)


def test_set_ret_vol_corr(dm):
    dm.set_ret_vol_corr(names=["SEB Hybrid", "High Yield"])
    assert dm.ret_vol.to_dict(as_series=False) == {
        "name": ["SEB Hybrid", "High Yield"],
        "long_name": [
            "SEB Hybrid Capital Bond Fund R-EUR",
            "SEB Global High Yield EUR",
        ],
        "exp_ret": [0.078, 0.064],
        "vol": [0.03771, 0.04526],
    }
    assert dm.corr.to_dict(as_series=False) == {
        "name": ["SEB Hybrid", "High Yield"],
        "SEB Hybrid": [1.0, 0.54],
        "High Yield": [0.54, 1.0],
    }

    assert_array_almost_equal(dm.get_vol(), np.array([0.03771, 0.04526]))
    assert_array_almost_equal(dm.get_ret(), np.array([0.078, 0.064]))
    assert_array_almost_equal(
        dm.get_covar(), np.array([[0.00142204, 0.00092165], [0.00092165, 0.00204847]])
    )


def test_get_min_max_ret(dm):
    r_min, r_max = dm.get_min_max_ret()
    assert r_min == 0.04
    assert r_max == 0.1


def test_names(dm):
    assert dm.names() == [
        "SEB Hybrid",
        "T-Bills",
        "Corps",
        "High Yield",
        "Equity",
        "Climate Focus",
        "Asset Selection",
    ]


def test_last_update(dm):
    assert dm.last_update().to_dict(as_series=False) == {
        "id": [1, 2, 3, 4, 5, 6, 7],
        "name": [
            "SEB Hybrid",
            "T-Bills",
            "Corps",
            "High Yield",
            "Equity",
            "Climate Focus",
            "Asset Selection",
        ],
        "yahoo": [
            "0P0001QM99.F",
            "0P0000XBX1.F",
            "0P00005Z2A.F",
            "0P0001CDNC.F",
            "0P000157DY.F",
            "0P0000ZW1Z.F",
            "0P0001CBFM.F",
        ],
        "min_date": [
            datetime.date(2023, 3, 30),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 3),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 2),
            datetime.date(2023, 1, 2),
        ],
        "max_date": [
            datetime.date(2024, 3, 15),
            datetime.date(2024, 3, 15),
            datetime.date(2024, 3, 18),
            datetime.date(2024, 3, 15),
            datetime.date(2024, 3, 15),
            datetime.date(2024, 3, 14),
            datetime.date(2024, 3, 15),
        ],
    }


def test_get_cumulative_rets_with_OPT(dm):
    c_rets = dm.get_cumulative_rets_with_OPT(
        names=["SEB Hybrid", "Corps"], w=np.array([0.6, 0.4])
    )
    assert c_rets.shape == (720, 3)
    assert c_rets[0, 2] == 0
    assert c_rets[-2:].with_columns(cs.by_dtype(pl.Float64).round(4)).to_dict(
        as_series=False
    ) == {
        "date": [datetime.date(2024, 3, 15), datetime.date(2024, 3, 18)],
        "name": ["OPTIMAL", "OPTIMAL"],
        "return": [0.1024, 0.1021],
    }
