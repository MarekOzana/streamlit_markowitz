"""
Unit tests on data_manager module

.. author:: Marek Ozana
.. date:: 2024-03
"""

import requests
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest
from numpy.testing import assert_array_almost_equal

from src.data_manager import DataManager
from src.data_manager import Updater


# Fixture to create a DataManager instance
@pytest.fixture
def dm():
    dm = DataManager(
        fund_tbl=Path("tests/data/t_fund.csv"),
        price_tbl=Path("tests/data/t_price.parquet"),
    )
    return dm


def test__init__(dm: DataManager):
    assert dm.fund_tbl == Path("tests/data/t_fund.csv")
    assert dm.price_tbl == Path("tests/data/t_price.parquet")
    assert dm.t_fund.shape == (7, 6)
    assert dm.t_fund.columns == [
        "id",
        "name",
        "long_name",
        "yahoo",
        "portf",
        "exp_ret",
    ]
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


def test_setup_session_with_proxy(dm: DataManager):
    with patch.dict("os.environ", {"YFIN_PROXY": "http://proxyserver:port"}):
        session = dm._setup_session()
        assert isinstance(session, requests.sessions.Session)
        assert session.proxies["http"] == "http://proxyserver:port"
        assert session.proxies["https"] == "http://proxyserver:port"


def test_setup_session_without_proxy(dm: DataManager):
    with patch.dict("os.environ", {}, clear=True):
        session = dm._setup_session()
        assert isinstance(session, requests.sessions.Session)
        assert not session.proxies


def test_download_data(dm: DataManager, caplog: pytest.LogCaptureFixture):
    session = requests.Session()
    f_row = {
        "name": "SEB Hybrid",
        "start_date": "2024-03-18",
        "yahoo": "yahoo.ticker",
        "id": 1,
    }

    # Create a mock DataFrame to be returned by yfinance.Ticker.history
    mock_ts = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "Close": [105, 106, 107],
            "High": [110, 111, 112],
            "Low": [95, 96, 97],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range(start="2024-03-18", periods=3),
    )

    # Patch yfinance.Ticker.history to return the mock DataFrame
    with patch("yfinance.Ticker.history", return_value=mock_ts):
        with caplog.at_level("DEBUG"):
            result = dm._download_data(session, f_row)

    # Verify the returned DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 3)  # 3 rows and 3 columns (fund_id, date, price)
    assert all(result["fund_id"] == f_row["id"])
    assert list(result["date"]) == list(mock_ts.index.date)
    assert list(result["price"]) == list(mock_ts["Close"])
    assert "Updating SEB Hybrid from 2024-03-18" in caplog.text


def test_download_data_empty_series(dm: DataManager, caplog: pytest.LogCaptureFixture):
    session = requests.Session()
    f_row = {
        "name": "SEB Hybrid",
        "start_date": "2024-03-18",
        "yahoo": "yahoo.ticker",
        "id": 1,
    }

    # Create an empty DataFrame to simulate no data returned by yfinance.Ticker.history
    mock_ts = pd.DataFrame(columns=["Open", "Close", "High", "Low", "Volume"])

    # Patch yfinance.Ticker.history to return the empty mock DataFrame
    with patch("yfinance.Ticker.history", return_value=mock_ts):
        with caplog.at_level("DEBUG"):
            result = dm._download_data(session, f_row)

    # Verify that the result is None since there's no data
    assert result is None
    # Check if the appropriate log message was generated
    assert "Updating SEB Hybrid from 2024-03-18" in caplog.text
    assert "No data for SEB Hybrid" in caplog.text


def test_update_from_yahoo(dm: DataManager, caplog: pytest.LogCaptureFixture):
    # Mock _setup_session and _download_data to isolate update_from_yahoo testing
    with patch.object(dm, "_setup_session") as m_setup:
        with patch.object(dm, "_download_data") as m_download:
            with patch.object(pl.DataFrame, "write_parquet") as m_write:
                m_setup.return_value = MagicMock()
                m_download.return_value = pl.DataFrame(
                    {
                        "fund_id": [1],
                        "date": [datetime.date(2023, 3, 15)],
                        "price": [115.0],
                    }
                ).with_columns(DataManager.PRICE_COLS)

                with caplog.at_level(level="DEBUG"):
                    dm.update_from_yahoo()
                # Assertions will depend on what you want to check, for example:
                m_setup.assert_called_once()
                m_download.assert_called()
                m_write.assert_called_once_with(dm.price_tbl)
                assert r"Saving data to tests" in caplog.text


def test_set_ret_vol_corr(dm: DataManager):
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


def test_get_min_max_ret(dm: DataManager):
    r_min, r_max = dm.get_min_max_ret()
    assert r_min == 0.04
    assert r_max == 0.1


def test_names(dm: DataManager):
    assert dm.names() == [
        "SEB Hybrid",
        "T-Bills",
        "Corps",
        "High Yield",
        "Equity",
        "Climate Focus",
        "Asset Selection",
    ]


def test_last_update(dm: DataManager):
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


def test_get_cumulative_rets_with_OPT(dm: DataManager):
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


class TestUpdater:
    @pytest.fixture
    def updater(self):
        f_name = Path("tests/data/M_Funds.csv")
        return Updater(f_name)

    def test_save_t_exp_table(self, updater: Updater, tmp_path: Path, caplog):
        o_path: Path = tmp_path / "test_t_exp.parquet"
        with caplog.at_level(level="DEBUG"):
            updater.save_t_exp_table(o_name=o_path)
        
        assert "Saving exposures to" in caplog.text
        # Verify the file exists
        assert o_path.exists()

        # Load the parquet file and verify its contents (optional)
        df = pl.read_parquet(o_path)
        assert df.shape == (8, 5)
        assert df.with_columns(cs.by_dtype(pl.Float64).round(2)).to_dict(
            as_series=False
        ) == {
            "portf": [
                "HYBRID",
                "HYBRID",
                "HYBRID",
                "HYBRID",
                "EURHYL HOLD",
                "EURHYL HOLD",
                "EURHYL HOLD",
                "EURHYL HOLD",
            ],
            "m_rating": ["BBB", "BBB", "BBB", "Cash", "BB", "BB", "B", "Cash"],
            "rating": ["BBB", "BBB-", "BBB-", "AA+", "BB+", "BB+", "B+", "AA+"],
            "ticker": [
                "DNBNO",
                "SEB",
                "RABOBK",
                "Cash",
                "KBCBB",
                "VOD",
                "TITIM",
                "Cash",
            ],
            "mv_pct": [0.03, 0.02, 0.01, 0.04, 0.01, 0.0, 0.01, 0.03],
        }

    def test_import_fund_info(self, updater: Updater):
        assert isinstance(updater.tbl, pl.LazyFrame)
        tbl = updater.tbl.collect()
        assert tbl.shape == (10, 23)
        assert tbl["rating"].to_list() == [
            "BBB-",
            "BBB",
            "BBB-",
            "BBB-",
            "AA+",
            "BB+",
            "BB+",
            "BB+",
            "B+",
            "AA+",
        ]
