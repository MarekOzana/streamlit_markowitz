"""
Utilities to get Yahoo data for quarterly Micro Finance optimization

.. author:: Marek Ozana
.. date:: 2024-08
"""

import logging

import polars as pl
import yfinance as yf

logger: logging.Logger = logging.getLogger(__name__)


def update_quarterly_data():
    """Update quarterly retuns for set of tickers
    Read data/MicroRets.csv for Micro FInance returns
    Read Major assets from Yahoo

    Create files
        data/micfin_q_rets.csv ... time series with quarterly returns
        data/micfin_exp_rets.csv ... expected annual returns
    """
    # 1. Get data from Yahoo
    tick2name = {
        "^GSPC": "S&P500",
        "FEZ": "EStoxx50",
        "IEAC.AS": "Euro Corp",
        "IHYA.L": "US High Yield",
        "EUNW.DE": "Euro High Yield",
        "EU13.L": "Euro Bills",
        "AT1.L": "AT1",
    }
    tickers = ",".join(tick2name.keys())
    yf_rets = (
        yf.download(tickers, start="2013-03-30", interval="1mo", progress=False)[
            "Adj Close"
        ]
        .rename(columns=tick2name)
        .resample("QE")
        .last()
        .pct_change()
        .dropna(how="all")
    )

    # POLARS implementation
    rets1 = pl.DataFrame(yf_rets.reset_index(names=["date"])).with_columns(
        pl.col("date").cast(pl.Date)
    )
    rets2 = pl.read_csv(
        "data/MicroRets.csv",
        schema_overrides={"date": pl.Date},
    )
    rets = rets1.join(rets2, on="date", how="left")

    # Save time series
    f_name = "data/micfin_q_rets.csv"
    rets.write_csv(f_name)
    logger.info(f"Saved to {f_name}")

    # Save expected returns
    exp_rets = rets.drop("date").mean() * 4
    exp_rets = exp_rets.transpose(
        include_header=True, header_name="ticker", column_names=["exp_ret"]
    )
    f_name = "data/micfin_exp_rets.csv"
    exp_rets.write_csv(f_name)
    logger.info(f"Saved to {f_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    update_quarterly_data()
