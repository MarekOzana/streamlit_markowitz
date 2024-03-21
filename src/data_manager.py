"""
Data Management Module
======================
Handle download and storage of historical time series and calculations
and caching of expected return, volatilities, correlations and covariances

the internal storage is done in polars.DataFrame
downloads are based on yfinance. if proxy is needed then set YFIN_PROXY env

.. author:: Marek Ozana
.. date:: 2024-03
"""

import logging
import os
from pathlib import Path
from datetime import date

import certifi
import numpy as np
import polars as pl
import polars.selectors as cs
import requests
import yfinance as yf

logger: logging.Logger = logging.getLogger(__name__)


class DataManager:
    FUND_COLS: list = [
        pl.col("id").cast(pl.Int16),
        pl.col("name").cast(pl.Utf8),
        pl.col("long_name").cast(pl.Utf8),
        pl.col("yahoo").cast(pl.Utf8),
        pl.col("exp_ret").cast(pl.Float64),
    ]
    PRICE_COLS: list = [
        pl.col("fund_id").cast(pl.Int16),
        pl.col("date").cast(pl.Date),
        pl.col("price").cast(pl.Float64),
    ]

    def __init__(
        self,
        fund_tbl: Path = Path("data/t_fund.csv"),
        price_tbl: Path = Path("data/t_price.parquet")
    ):
        self.fund_tbl:Path = Path(fund_tbl)
        self.price_tbl:Path = Path(price_tbl)
        self.t_fund = pl.read_csv(fund_tbl).with_columns(self.FUND_COLS)
        self.t_price = pl.read_parquet(price_tbl).with_columns(self.PRICE_COLS)
        self.set_ret_vol_corr(self.names())  # initialize

    def update_from_yahoo(self) -> None:
        """Update time series for all tickers from yahoo finance"""
        session = requests.Session()
        if (yfin_proxy := os.getenv("YFIN_PROXY")) is not None:
            logger.debug(f"Using proxy: {yfin_proxy}")
            proxies = {
                "http": yfin_proxy,
                "https": yfin_proxy,
            }
            session.proxies.update(proxies)
            session.verify = certifi.where()  # append SEB's .pem to cacert.pem
        else:
            logger.debug("Not using proxy")

        tbl = self.last_update().with_columns(
            (
                pl.col("max_date").fill_null(date(2022, 12, 31)) + pl.duration(days=1)
            ).alias("start_date")
        )
        for f_row in tbl.rows(named=True):
            logger.info(f"Updating {f_row['name']} from {f_row['start_date']}")

            # Download data
            ts = yf.Ticker(ticker=f_row["yahoo"], session=session).history(
                start=f_row["start_date"], interval="1d"
            )
            if len(ts) == 0:
                continue
            # Append to existing table
            new_prices = pl.DataFrame(
                {
                    "fund_id": f_row["id"],
                    "date": list(ts.index.date),
                    "price": ts["Close"],
                }
            ).with_columns(self.PRICE_COLS)
            self.t_price = pl.concat([self.t_price, new_prices])

        # Save the data
        logger.info(f"Saving data to {self.price_tbl}")
        self.t_price.write_parquet(self.price_tbl)

    def set_ret_vol_corr(self, names: list) -> tuple:
        """Calculate and set vol, corr and exp_returns given list of 'names'

        Returns
        -------
        ret_vol:pl.DataFrame(name, ret, vol)
        corr: pl.DataFrame(name, correlations)
        """
        daily_rets = self.get_daily_rets(names=names).select(cs.by_dtype(pl.Float64))
        vols = daily_rets.select(pl.all().std().mul(np.sqrt(252)).round(5)).transpose(
            include_header=True, header_name="name", column_names=["vol"]
        )
        self.ret_vol = self.t_fund.select(pl.col("name", "long_name", "exp_ret")).join(
            vols, on="name"
        )
        corr = daily_rets.drop_nulls().corr().select(pl.all().round(3))
        self.corr = corr.insert_column(0, pl.Series("name", corr.columns))
        # assert the same order
        assert self.corr["name"].to_list() == self.ret_vol["name"].to_list()

    def get_vol(self) -> np.array:
        return self.ret_vol["vol"].to_numpy()

    def get_ret(self) -> np.array:
        return self.ret_vol["exp_ret"].to_numpy()

    def get_covar(self) -> np.array:
        """Calculate covariance matrix"""
        logger.debug("Calculating covariance matrix")
        vol = np.diag(self.get_vol())
        corr = self.corr.select(cs.exclude("name")).to_numpy()
        cov = vol @ corr @ vol
        return cov

    def get_min_max_ret(self):
        """Return min and max expected return"""
        r_min, r_max = self.ret_vol.select(
            pl.col("exp_ret").min().alias("r_min"),
            pl.col("exp_ret").max().alias("r_max"),
        ).rows()[0]
        return r_min, r_max

    def names(self) -> list[str]:
        """Get list of all names"""
        return self.t_fund["name"].to_list()

    def last_update(self) -> pl.DataFrame:
        """generate table with fund_id, min_date, max_date, name and yahoo"""
        tbl = self.t_fund.select("id", "name", "yahoo").join(
            self.t_price.group_by("fund_id").agg(
                pl.col("date").min().alias("min_date"),
                pl.col("date").max().alias("max_date"),
            ),
            how="left",
            left_on="id",
            right_on="fund_id",
        )
        return tbl

    def get_daily_rets(self, names: list) -> pl.DataFrame:
        """get wide dataframe with daily returns"""
        nav_data = (
            self.t_price.join(
                self.t_fund.select(pl.col("id", "name")),
                left_on="fund_id",
                right_on="id",
            )
            .filter(pl.col("name").is_in(names))
            .pivot(index="date", columns="name", values="price")
            .sort("date")
            .fill_null(strategy="forward")
        )
        daily_rets = nav_data.with_columns(cs.by_dtype(pl.Float64).pct_change())
        return daily_rets

    def get_cumulative_rets_with_OPT(self, names: list, w: np.array) -> pl.DataFrame:
        """Return long dataframe with cumulative returns for each ticker
        and for optimal porfolio defined by weights array 'w'
        """
        # Cumulative returns chart
        start_dt: date = date(2023, 3, 30)
        # daily returns including optimal portfolio in wide format
        r_cum = (
            (
                self.get_daily_rets(names=names)
                .filter(pl.col("date") >= start_dt)
                .with_columns(
                    pl.sum_horizontal(
                        [pl.col(col) * wgt for col, wgt in zip(names, w)]
                    ).alias("OPTIMAL")
                )
                .fill_null(0)
            )
            .with_columns(cs.by_dtype(pl.Float64).add(1).cum_prod().sub(1))
            .melt(id_vars="date", variable_name="name", value_name="return")
            # set start point to 0%
            .with_columns(
                pl.when(pl.col("date") == pl.col("date").min())
                .then(0)
                .otherwise(pl.col("return"))
                .alias("return")
            )
        )
        return r_cum
