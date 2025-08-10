"""
Updated unit tests on data_manager module after yfinance session refactor.

.. author:: Marek Ozana
.. date:: 2025-08
"""

import sys
import types
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
import src.data_manager


# Fixture to create a DataManager instance
@pytest.fixture
def dm():
    dm = DataManager(
        fund_tbl=Path("tests/data/t_fund.csv"),
        price_tbl=Path("tests/data/t_price.parquet"),
        exp_tbl=Path("tests/data/t_exp.parquet"),
    )
    return dm


def test__init__(dm: DataManager):
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
    assert isinstance(dm.t_exp, pl.DataFrame)
    assert dm.t_exp.shape == (139, 6)
    assert dm.t_exp.columns == [
        "portf",
        "m_rating",
        "rating",
        "ticker",
        "mv_pct",
        "date",
    ]


def test_yf_session_with_proxy(dm: DataManager, monkeypatch):
    """_yf_session_or_none returns a curl_cffi-like session and applies proxies."""

    # Dummy curl_cffi.requests.Session to avoid real dependency
    class DummySession:
        def __init__(self, impersonate=None):
            self.proxies = {}
            self.verify = None

    dummy_requests = types.SimpleNamespace(Session=DummySession)
    dummy_module = types.SimpleNamespace(requests=dummy_requests)

    monkeypatch.setenv("YFIN_PROXY", "http://proxyserver:port")
    # Inject dummy curl_cffi into sys.modules
    monkeypatch.setitem(sys.modules, "curl_cffi", dummy_module)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", dummy_requests)

    s = dm._yf_session_or_none()
    assert isinstance(s, DummySession)
    assert s.proxies["http"] == "http://proxyserver:port"
    assert s.proxies["https"] == "http://proxyserver:port"
    assert s.verify  # certifi path set


def test_yf_session_without_proxy(dm: DataManager, monkeypatch):
    monkeypatch.delenv("YFIN_PROXY", raising=False)
    session = dm._yf_session_or_none()
    assert session is None


def test_download_data(dm: DataManager, caplog: pytest.LogCaptureFixture):
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
            result = dm._download_data(f_row)

    # Verify the returned DataFrame
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 3)  # 3 rows and 3 columns (fund_id, date, price)
    assert all(result["fund_id"] == f_row["id"])
    assert list(result["date"]) == list(mock_ts.index.date)
    assert list(result["price"]) == list(mock_ts["Close"])
    assert "Updating SEB Hybrid from 2024-03-18" in caplog.text


def test_download_data_empty_series(dm: DataManager, caplog: pytest.LogCaptureFixture):
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
            result = dm._download_data(f_row)

    # Verify that the result is None since there's no data
    assert result is None
    # Check if the appropriate log message was generated
    assert "Updating SEB Hybrid from 2024-03-18" in caplog.text
    assert "No data for SEB Hybrid" in caplog.text


def test_update_from_yahoo(dm: DataManager, caplog: pytest.LogCaptureFixture):
    # Mock _download_data to isolate update_from_yahoo testing
    with patch.object(dm, "_download_data") as m_download:
        with patch.object(pl.DataFrame, "write_parquet") as m_write:
            m_download.return_value = pl.DataFrame(
                {
                    "fund_id": [1],
                    "date": [datetime.date(2023, 3, 15)],
                    "price": [115.0],
                }
            ).with_columns(DataManager.PRICE_COLS)
            # Use a MagicMock for the callback to check calls
            m_callback = MagicMock()
            with caplog.at_level(level="DEBUG"):
                dm.update_from_yahoo(callback=m_callback)
            # Assertions:
            m_download.assert_called()
            m_write.assert_called_once_with(dm.price_tbl)
            assert r"Saving data to tests" in caplog.text
            assert m_callback.call_count == 7
            m_callback.assert_any_call(1.0 / 7.0)


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


def test_get_cumulative_rets_and_dd(dm: DataManager):
    r_cum = dm.get_cumulative_rets_and_dd(name="SEB Hybrid")
    assert isinstance(r_cum, pl.DataFrame)
    assert r_cum.columns == ["date", "SEB Hybrid", "DrawDown"]
    assert r_cum.shape == (234, 3)
    assert round(r_cum.item(-1, "SEB Hybrid"), 3) == 0.125


def test_get_monthly_perf(dm: DataManager):
    m_tbl = dm.get_monthly_perf(name="SEB Hybrid")
    assert m_tbl.shape == (2, 14)
    assert m_tbl.with_columns(cs.by_dtype(pl.Float64).round(4).fill_null(0)).to_dict(
        as_series=False
    ) == {
        "Year": [2023, 2024],
        "Jan": [0.0, 0.0102],
        "Feb": [0.0, 0.0021],
        "Mar": [0.0019, 0.0127],
        "Apr": [-0.0056, 0.0],
        "May": [0.0075, 0.0],
        "Jun": [0.0053, 0.0],
        "Jul": [0.0245, 0.0],
        "Aug": [-0.007, 0.0],
        "Sep": [-0.0021, 0.0],
        "Oct": [0.0015, 0.0],
        "Nov": [0.0358, 0.0],
        "Dec": [0.0323, 0.0],
        "YTD": [0.097, 0.0252],
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


def test_get_fund_exposures(dm: DataManager):
    df = dm.get_fund_exposures(name="SEB Hybrid")
    assert df.shape == (58, 5)
    assert df[-2].to_dict(as_series=False) == {
        "name": ["SEB Hybrid"],
        "date": [datetime.date(2024, 3, 27)],
        "m_rating": ["BB"],
        "ticker": ["LLOYDS"],
        "mv_pct": [0.0021],
    }


class TestUpdater:
    @pytest.fixture
    def updater(self):
        f_name = Path("tests/data/M_Funds.csv")
        return Updater(f_name)

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

    def test__extract_report_date(self, updater: Updater):
        assert updater.as_of == datetime.date(2024, 3, 27)

    def test_save_t_exp_table(
        self, updater: Updater, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        o_path: Path = tmp_path / "test_t_exp.parquet"
        with caplog.at_level(level="DEBUG"):
            updater.save_t_exp_table(o_name=o_path)

        assert "Saving (8, 6) exposures to" in caplog.text
        # Verify the file exists
        assert o_path.exists()

        # Load the parquet file and verify its contents (optional)
        df = pl.read_parquet(o_path)
        assert df.shape == (8, 6)
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
            "date": [
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
            ],
        }

    def test_save_t_keyfigures_table(
        self, updater: Updater, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        o_path: Path = tmp_path / "test_t_keyfigures.parquet"
        with caplog.at_level(level="DEBUG"):
            updater.save_t_keyfigures_table(o_name=o_path)

        assert "Saving (8, 4) key figures to" in caplog.text
        # Verify the file exists
        assert o_path.exists()

        # Load the parquet file and verify its contents (optional)
        df = pl.read_parquet(o_path)
        assert df.shape == (8, 4)
        assert df.columns == ["date", "portf", "key", "value"]
        assert df.to_dict(as_series=False) == {
            "date": [
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
                datetime.date(2024, 3, 27),
            ],
            "portf": [
                "EURHYL HOLD",
                "EURHYL HOLD",
                "EURHYL HOLD",
                "EURHYL HOLD",
                "HYBRID",
                "HYBRID",
                "HYBRID",
                "HYBRID",
            ],
            "key": ["oas", "ytc", "ytw", "zspread", "oas", "ytc", "ytw", "zspread"],
            "value": [314.27, 218.91, 6.88, 369.9, 329.62, 7.34, 7.18, 346.97],
        }


class TestCLI:
    """Unit tests on command line interface for Updater"""

    @patch("src.data_manager.Updater")
    def test_main(self, m_updater, caplog: pytest.LogCaptureFixture):
        m_instance = MagicMock()
        m_updater.return_value = m_instance
        test_args = ["data_manager.py", "funds", "-f", "M_Funds.csv"]
        with patch("sys.argv", test_args):
            src.data_manager.main()
        # Check if Updater was called correctly
        m_updater.assert_called_once_with("M_Funds.csv")

        # Verify that save_t_exp_table and save_t_keyfigures_table were called
        m_instance.save_t_exp_table.assert_called_once_with(
            o_name=Path("data/t_exp.parquet")
        )
        m_instance.save_t_keyfigures_table.assert_called_once_with(
            o_name=Path("data/t_keyfigures.parquet")
        )

        # Check logging message
        assert "DONE" in caplog.text
