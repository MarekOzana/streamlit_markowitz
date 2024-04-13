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

import argparse
import logging
import os
from pathlib import Path
from datetime import date
import datetime
import re
from typing import Optional
import calendar

from filelock import FileLock
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
        pl.col("portf").cast(pl.Utf8),
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
        price_tbl: Path = Path("data/t_price.parquet"),
        exp_tbl: Path = Path("data/t_exp.parquet"),
    ):
        self.price_tbl: Path = Path(price_tbl)
        self.lock: FileLock = FileLock(self.price_tbl.with_suffix(".lock"))

        self.t_fund = pl.read_csv(fund_tbl).with_columns(self.FUND_COLS)
        self.t_exp = pl.read_parquet(exp_tbl)
        with self.lock:
            self.t_price = pl.read_parquet(price_tbl).with_columns(self.PRICE_COLS)
        self.set_ret_vol_corr(self.names())  # initialize

    def _setup_session(self) -> requests.Session:
        """
        Initialize and configure an HTTP session for fetching data from Yahoo Finance.

        This method sets up a requests session, applying a proxy configuration if the
        YFIN_PROXY environment variable is set. If the proxy is set, it configures the
        session to use this proxy for HTTP and HTTPS requests. Additionally, it
        configures the session to use the appropriate SSL certification based
        on the certifi library's default location.

        Returns:
            requests.Session: A configured requests.Session object ready for use in data
            fetching, with or without proxy settings applied.
        """
        session = requests.Session()
        yfin_proxy = os.getenv("YFIN_PROXY")
        if yfin_proxy:
            logger.debug(f"Using proxy: {yfin_proxy}")
            session.proxies.update({"http": yfin_proxy, "https": yfin_proxy})
            session.verify = certifi.where()
        else:
            logger.debug("Not using proxy")
        return session

    def _download_data(self, session: requests.Session, f_row: dict) -> pl.DataFrame:
        """
        Download historical stock data from Yahoo Finance for a specific fund.

        This method takes a session and a fund row, downloads historical stock data
        from the specified start date using Yahoo Finance, and returns a DataFrame
        with the new prices.

        Parameters:
            session (requests.Session): The HTTP session to be used forAPI calls.
            f_row (dict): A dictionary representing a row from the funds table. It
                        includes the following keys:
                        - 'name': The name of the fund.
                        - 'yahoo': The ticker symbol (Yahoo Finance).
                        - 'start_date': The date from which to start fetching the data.
                        - 'id': The unique identifier of the fund.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the downloaded price data.
                            The DataFrame includes the following columns:
                            'fund_id', 'date', and 'price'.
                            If no data is downloaded, None is returned.

        Note:
            If the historical data for the specified fund is empty
        """
        logger.info(f"Updating {f_row['name']} from {f_row['start_date']}")
        yf_ticker = yf.Ticker(ticker=f_row["yahoo"], session=session)
        ts = yf_ticker.history(start=f_row["start_date"], interval="1d")
        if len(ts) == 0:
            logger.debug(f"No data for {f_row['name']}")
            return None
        new_prices = pl.DataFrame(
            {"fund_id": f_row["id"], "date": list(ts.index.date), "price": ts["Close"]}
        ).with_columns(self.PRICE_COLS)
        return new_prices

    def update_from_yahoo(self, callback=None) -> None:
        """Update time series for all tickers from yahoo finance"""
        session = self._setup_session()

        tbl = self.last_update().with_columns(
            (
                pl.col("max_date").fill_null(date(2022, 12, 31)) + pl.duration(days=1)
            ).alias("start_date")
        )
        is_updated: bool = False
        for i, f_row in enumerate(tbl.rows(named=True)):
            new_prices = self._download_data(session, f_row)
            if new_prices is not None:
                self.t_price = pl.concat([self.t_price, new_prices])
                is_updated = True
            if callback:
                progress = (i + 1) / len(tbl)
                callback(progress)

        # Save the data
        if is_updated:
            logger.info(f"Saving data to {self.price_tbl}")
            with self.lock:
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

    def get_cumulative_rets_and_dd(self, name: str) -> pl.DataFrame:
        """Return time series with cumulative returns and draw-down for single security"""
        df: pl.DataFrame = (
            self.get_daily_rets(names=[name])
            .fill_null(0)
            .with_columns(pl.col(name).add(1).cum_prod().sub(1))
            .with_columns(pl.col(name).add(1).cum_max().alias("prev_peaks"))
            .with_columns(
                pl.col(name)
                .add(1)
                .sub(pl.col("prev_peaks"))
                .truediv(pl.col("prev_peaks"))
                .alias("DrawDown")
            )
            .select(["date", name, "DrawDown"])
        )
        return df

    def get_cumulative_rets_with_OPT(self, names: list, w: np.array) -> pl.DataFrame:
        """Return long dataframe with cumulative returns for each ticker
        and for optimal porfolio defined by weights array 'w'
        NOTE: the start date is fixed ot 30th of March
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

    def get_monthly_perf(self, name: str) -> pl.DataFrame:
        """
        Retrieves the monthly performance as a DataFrame with years in the rows,
        months in the columns, and Year-to-Date (YTD) returns in the last column.

        Parameters:
            name (str): The name of the asset for which to calculate performance.

        Returns:
            pl.DataFrame: A DataFrame with perf metrics organized by month and year.
        """
        # Get daily returns for the specified name
        d_rets: pl.DataFrame = self.get_daily_rets(names=[name])

        # Calculate monthly returns
        m_rets: pl.DataFrame = (
            d_rets.group_by(
                [
                    pl.col("date").dt.year().alias("Year"),
                    pl.col("date").dt.month().alias("Month"),
                ]
            )
            .agg((pl.col(name).add(1).product() - 1).alias("ret"))
            .sort(["Year", "Month"])
        )

        # Calculate Year-to-Date returns
        y_rets: pl.DataFrame = m_rets.group_by("Year").agg(
            (pl.col("ret").add(1).product() - 1).alias("YTD")
        )

        # Pivot table to transform data for easier year/month viewing
        m_tbl: pl.DataFrame = (
            m_rets.pivot(values="ret", index="Year", columns="Month")
            .sort("Year")
            .select(["Year"] + [str(i) for i in range(1, 13)])
        )

        # Join the YTD data
        m_tbl = m_tbl.join(y_rets, on="Year")

        # Rename columns from month numbers to month abbreviations
        month_map = {str(i): calendar.month_abbr[i] for i in range(1, 13)}
        m_tbl = m_tbl.rename(month_map)

        return m_tbl

    def get_fund_exposures(self, name: str) -> pl.DataFrame:
        """Get fund exposures
        Params
        ------
        name: str
            fund name

        Returns
        -------
        df: pl.DataFrame (m_rating, ticker, mv_pct)
        """
        df = (
            self.t_exp.join(self.t_fund, on="portf")
            .filter(pl.col("name") == name)
            .select(["name", "date", "m_rating", "ticker", "mv_pct"])
            .with_columns(pl.col("m_rating").fill_null("NR"))
        )
        return df


class Updater:
    PREP2COLS: dict[str, str] = {
        "Ticker": "ticker",
        "Cpn": "cpn",
        "Crncy": "crncy",
        "Next Call Date": "next_call",
        "Cntry": "country",
        "Yield to Call": "ytc",
        "Yield to Worst": "ytw",
        "Z-Spd": "zspread",
        "OAS": "oas",
        "S-OAS": "s_oas",
        "OAD": "oad",
        "OASD": "oasd",
        "DTS": "dts",
        "Index Rtg": "ix_rtg",
        "S&P": "rtg_sp",
        "Moody's": "rtg_moody",
        "Fitch": "rtg_fitch",
        "BB Comp": "rtg_bb",
        "Market Value (%)": "mv_pct",
        "Book Port": "portf",
    }

    MOODY_TO_SP: dict[str, str] = {
        "AAA": "AAA",
        "AA1": "AA+",
        "AA2": "AA",
        "AA3": "AA-",
        "A1": "A+",
        "A2": "A",
        "A3": "A-",
        "BAA1": "BBB+",
        "BAA2": "BBB",
        "BAA3": "BBB-",
        "BA1": "BB+",
        "BA2": "BB",
        "BA3": "BB-",
        "B1": "B+",
        "B2": "B",
        "B3": "B-",
        "CAA1": "CCC+",
        "CAA2": "CCC",
        "CAA3": "CCC-",
        "CA": "CC",
        "C": "C",
        "D": "D",
    }

    RATING_TO_MRATING: dict[str, str] = {
        "AAA": "AAA",
        "AA+": "AA",
        "AA": "AA",
        "AA-": "AA",
        "A+": "A",
        "A": "A",
        "A-": "A",
        "BBB+": "BBB",
        "BBB": "BBB",
        "BBB-": "BBB",
        "BB+": "BB",
        "BB": "BB",
        "BB-": "BB",
        "B+": "B",
        "B": "B",
        "B-": "B",
        "CCC+": "CCC",
        "CCC": "CCC",
        "CCC-": "CCC",
        "CC": "CC",
        "C": "C",
        "D": "D",
    }

    RATING_TYPE: pl.Enum = pl.Enum(
        [
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "CCC+",
            "CCC",
            "CCC-",
            "CC",
            "C",
            "D",
        ]
    )

    MRATING_TYPE: pl.Enum = pl.Enum(
        [
            "AAA",
            "AA",
            "A",
            "BBB",
            "BB",
            "B",
            "CCC",
            "CC",
            "C",
            "D",
            "NR",
            "Cash",
        ]
    )

    def __init__(self, f_name: Path) -> None:
        """
        Parse external CSV files and prepare tables

        Usage
        -----
        >>> u = Updater(f_name='tests/data/M_Funds.csv')
        >>> u.save_t_exp_table(o_name='data/t_exp.parquet')
        """
        self.as_of: date = self._extract_report_date(f_name)
        logger.info(f"Parsing {f_name} as of {self.as_of:%F}")
        self.tbl: pl.LazyFrame = self._import_fund_info(f_name)
        # TODO: insert key figures and date

    def _extract_report_date(self, f_name: Path) -> Optional[date]:
        """
        Extracts the report date from a text file.

        This function searches for a date in the format M/D/YYYY that follows
        the phrase "Report Period" in the provided text file.

        Parameters:
        f_name (Path): The path to the text file from which the date is to be extracted.

        Returns:
        Optional[date]: The extracted date as a datetime.date object or None
        """

        # Define the regex pattern to find the date in the format M/D/YYYY
        date_pattern = r"Report Period.*?(\d{1,2}/\d{1,2}/\d{4})"

        # Open and read the content of the file
        with open(f_name, "r") as file:
            content = file.read()

            # Search for the date pattern in the file content
            match = re.search(date_pattern, content)

            # If a match is found, convert the matched string to a date object
            if match:
                return datetime.datetime.strptime(match.group(1), "%m/%d/%Y").date()

        # Return None if no date is found
        return None

    def _import_fund_info(self, f_name: Path) -> pl.LazyFrame:
        """
        Import Fund exposures and key figures from csv file
        """
        tbl = (
            pl.scan_csv(
                f_name,
                has_header=True,
                skip_rows=7,
                dtypes={"Next Call Date": pl.Date},
                null_values=[""],
                new_columns=[
                    "id",
                ],
            )
            .rename(self.PREP2COLS)
            .filter(pl.col("mv_pct").is_not_null())
            .with_columns(
                pl.col("id").str.strip_chars_start(),
                pl.col("ix_rtg")
                .str.extract(r"^([ABCDabcd\d]+\+?-?)")
                .str.to_uppercase(),
                pl.col("rtg_moody")
                .str.extract(r"^([ABCDabcd\d]+\+?-?)")
                .str.to_uppercase(),
                pl.col("rtg_sp").str.extract(r"^([ABCDabcd\d]+\+?-?)"),
                pl.col("rtg_fitch").str.extract(r"^([ABCDabcd\d]+\+?-?)"),
            )
            .with_columns(
                # rating = ix -> moodys -> sp -> fitch
                pl.col("ix_rtg")
                .fill_null(pl.col("rtg_moody"))
                .map_elements(lambda x: self.MOODY_TO_SP.get(x))
                .fill_null(pl.col("rtg_sp"))
                .fill_null(pl.col("rtg_fitch"))
                .cast(self.RATING_TYPE)
                .alias("rating"),
            )
            .with_columns(
                pl.col("rating")
                .map_elements(lambda x: self.RATING_TO_MRATING.get(x))
                .cast(self.MRATING_TYPE)
                .alias("m_rating")
            )
            .with_columns(
                # Set ticker='Cash' for cash
                pl.when(pl.col("id").str.contains("Not Classified"))
                .then(pl.lit("Cash"))
                .otherwise(pl.col("ticker"))
                .alias("ticker"),
                # Set rating='AA?' for cash
                pl.when(pl.col("id").str.contains("Not Classified"))
                .then(pl.lit("AA+"))
                .otherwise(pl.col("rating"))
                .alias("rating"),
                # Set m_rating='Cash' for cash
                pl.when(pl.col("id").str.contains("Not Classified"))
                .then(pl.lit("Cash"))
                .otherwise(pl.col("m_rating"))
                .alias("m_rating"),
            )
        )
        return tbl

    def save_t_exp_table(self, o_name: Path = Path("data/t_exp.parquet")) -> None:
        """Create and save exposure table to o_name"""
        df_exp: pl.DataFrame = (
            self.tbl.group_by(["portf", "m_rating", "rating", "ticker"])
            .agg(pl.col("mv_pct").sum().mul(0.01))
            .filter(pl.col("ticker").is_not_null())  # Remove totals
            .sort(
                "portf",
                "m_rating",
                "rating",
                "mv_pct",
                descending=[True, False, False, True],
                nulls_last=True,
            )
            .with_columns(pl.lit(self.as_of).alias("date"))  # add as of date
            .collect()
        )
        logger.info(f"Saving {df_exp.shape} exposures to {o_name}")
        df_exp.write_parquet(o_name)

    def save_t_keyfigures_table(
        self, o_name: Path = Path("data/t_keyfigures.parquet")
    ) -> None:
        """Create and save Key Figures to o_name"""
        df_kf: pl.DataFrame = (
            self.tbl.filter(pl.col("ticker").is_null())
            .select(["portf", "ytc", "ytw", "oas", "zspread"])
            .with_columns(pl.lit(self.as_of).alias("date"))
            .melt(id_vars=["date", "portf"], variable_name="key")
            .with_columns(pl.col("value").cast(pl.Float64))
            .sort(by=["date", "portf", "key"])
            .collect()
        )
        logger.info(f"Saving {df_kf.shape} key figures to {o_name}")
        df_kf.write_parquet(o_name)


def main():
    parser = argparse.ArgumentParser(description="Update Exposures and KeyFigures")
    parser.add_argument("-f", "--file", required=True, help="M_Funds.csv file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    u = Updater(args.file)
    u.save_t_exp_table(o_name=Path("data/t_exp.parquet"))
    u.save_t_keyfigures_table(o_name=Path("data/t_keyfigures.parquet"))
    logger.info("DONE")


if __name__ == "__main__":
    main()
