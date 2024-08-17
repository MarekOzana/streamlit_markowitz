"""
Utilities to get data from Yahoo for quarterly optimization

.. author:: Marek Ozana
.. date:: 2024-08
"""

import yfinance as yf
import polars as pl


def update_quarterly_data():
    """Update quarterly retuns for set of tickers"""
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
    rets = rets1.join(rets2, on="date", how='left')
    
    f_name = "data/quarterly_rets.csv"
    rets.write_csv("data/quarterly_rets.csv")
    print(f"Saved to {f_name}")
