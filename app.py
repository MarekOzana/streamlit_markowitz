"""
Markowitz Optimization on Funds
author: 2024-03 Marek Ozana
"""

import logging
import pathlib

import numpy as np
import polars as pl
import polars.selectors as cs
import streamlit as st
from scipy.stats import norm

import src.charts as charts
import src.optimization as opt
from src.data_manager import DataManager

st.set_page_config(
    page_title="Markowitz Optimizer", layout="wide", page_icon="data/icon.png"
)

logger: logging.Logger = logging.getLogger(__name__)


def get_params(db: DataManager) -> float:
    """get tickers  and set them to DataManger, get r_min

    Returns
    -------
    r_min: float
    """
    # get r_min
    r_minimum, r_maximum = db.get_min_max_ret()
    with st.container(border=True):
        st.subheader("Try to change required return")
        r_min = (
            st.slider(
                label="Required Return",
                min_value=r_minimum * 100,
                max_value=r_maximum * 100,
                value=(r_minimum + r_maximum) / 2.0 * 100,
                format="%.1f%%",
                help="Constraint on required minimum return",
            )
            / 100
        )
    # get tickers
    tickers = st.multiselect(
        label="Tickers",
        options=db.names(),
        default=db.names()[:-1],
        help="Select Names",
    )
    if not tickers:
        tickers = db.names()
        st.warning("At least one ticker must be selected...")

    # Set tickers
    if ("tickers" not in st.session_state) or (tickers != st.session_state["tickers"]):
        st.session_state["tickers"] = tickers
        db.set_ret_vol_corr(tickers)

    st.divider()
    with st.popover("How to Use the App"):
        st.markdown(pathlib.Path("data/usage.txt").read_text())

    return r_min


def create_main_tab(db: DataManager, r_min: float) -> None:
    """Create main TAB with optimization charts and portfolio Weights"""
    # Optimize
    w, r_opt, vol_opt = opt.find_min_var_portfolio(db.get_ret(), db.get_covar(), r_min)

    # Create Scatter chart and Portfolio Composition Chart
    g_data = pl.DataFrame(
        {
            "name": db.ret_vol["name"].to_list() + ["OPTIMAL"],
            "vols": db.get_vol().tolist() + [vol_opt],
            "rets": db.get_ret().tolist() + [r_opt],
            "w_opt": w.tolist() + [1],
        }
    )
    f_sc = charts.create_scatter_chart(g_data)
    f_w = charts.create_portf_weights_chart(g_data)
    title = f"Optimal Portfolio: r={r_opt:0.1%}, vol={vol_opt:0.1%}"
    col1, col2 = st.columns([1.5, 1])
    col1.altair_chart(f_sc, use_container_width=True)
    col2.altair_chart(f_w.properties(title=title, height=350), use_container_width=True)
    st.markdown(
        f"""
        ### Optimal Portfolio Statistics
        * **Expected Return in 1y** = {r_opt:0.1%}
        * **Expected volatility** = {vol_opt:0.1%}
        * 95%-prob Lowest Expected Return in 1y = {(r_opt - norm.ppf(0.95)*vol_opt):0.1%}
        * 99%-prob Lowest Expected Return in 1y = {(r_opt - norm.ppf(0.99)*vol_opt):0.1%}

        **Probability of Negative Returns:**
        * in 1 month = {norm.cdf(0, r_opt/12.0, scale=vol_opt*np.sqrt(1/12.0)):0.0%}
        * in 1 quarter = {norm.cdf(0, 3*r_opt/12.0, scale=vol_opt*np.sqrt(3/12.0)):0.0%}
        * in 1 year  = {norm.cdf(0, r_opt, scale=vol_opt):0.1%}
        * in 2 years = {norm.cdf(0, 2*r_opt, scale=vol_opt*np.sqrt(2)):0.2%}
        """
    )

    # Cumulative returns chart
    r_cum = db.get_cumulative_rets_with_OPT(db.ret_vol["name"].to_list(), w)
    fig = charts.create_cum_ret_chart(r_cum)
    st.altair_chart(fig, use_container_width=True)

    # Create chart with Prob of negative returns
    p_fig = charts.create_prob_of_neg_chart(r_ann=r_opt, vol_ann=vol_opt, n=36)
    st.altair_chart(p_fig, use_container_width=True)


def create_edit_assumptions_tab(db: DataManager) -> None:
    with st.form("Edit Assumptions", border=True):
        ret_vol = st.data_editor(
            db.ret_vol.with_columns(cs.by_dtype(pl.Float64).mul(100))
            .to_pandas()
            .set_index("name"),
            column_config={
                "exp_ret": st.column_config.NumberColumn(
                    "Exp Return [%]", format="%0.1f %%"
                ),
                "vol": st.column_config.NumberColumn(
                    "Volatility [%]", format="%0.1f %%"
                ),
            },
        )

        corr = st.data_editor(
            db.corr.to_pandas().set_index("name") * 100,
            column_config={
                col: st.column_config.NumberColumn(
                    format="%.0f%%", min_value=-100, max_value=100
                )
                for col in db.corr["name"].to_list()
            },
        ).div(100)

        if st.form_submit_button("Update Return / Vol / Correlations"):
            st.info("Updateding values")
            db.ret_vol = pl.DataFrame(ret_vol.reset_index()).with_columns(
                cs.by_dtype(pl.Float64).mul(0.01)
            )
            db.corr = pl.DataFrame(corr.reset_index())
            st.rerun()  # Make sure the values are updated

    st.divider()
    st.info(f"Last Price Update: {db.last_update()['max_date'].max():%F}")
    if st.button("Update Data from Yahoo"):
        my_bar = st.progress(0, text="Updating from Yahoo")
        db.update_from_yahoo(callback=lambda x: my_bar.progress(x))
        st.rerun()  # Make sure the values are updated


def create_fund_info_tab(db):
    col1, col2 = st.columns([4, 1])
    with col2:
        # Select & Statistics
        name = st.selectbox("Fund", options=db.names(), label_visibility="collapsed")
    with col1:
        # Performance and DrawDowns
        df = db.get_cumulative_rets_and_dd(name=name)
        fig = charts.create_cumul_ret_with_drawdown_chart(df)
        col1.altair_chart(fig, use_container_width=True)

    # MOnthly Performance
    m_tbl: pl.DataFrame = db.get_monthly_perf(name=name)
    m_style = (
        m_tbl.with_columns(pl.col("Year").cast(str))
        .to_pandas()
        .set_index("Year")
        .style.format("{:0.2%}", na_rep="")
    )
    st.dataframe(m_style)

    # Rating and Ticker Exposures
    df = db.get_fund_exposures(name=name)
    if len(df) > 0:
        fig = charts.create_exp_chart(df)
        st.altair_chart(fig, use_container_width=True)
    else:
        st.info(f"Exposure info NOT available for {name}")


@st.cache_resource
def get_db() -> DataManager:
    db = DataManager()
    return db


def main() -> None:
    st.title("Markowitz Optimization")
    db = get_db()

    with st.sidebar:
        st.title("Parameters")
        r_min = get_params(db)

    tab_main, tab_data, tab_fund = st.tabs(["Optimal Portfolio", "Data", "Fund Info"])
    with tab_main:
        create_main_tab(db, r_min)

    with tab_data:
        create_edit_assumptions_tab(db)

    with tab_fund:
        create_fund_info_tab(db)

    st.divider()
    st.caption(pathlib.Path("data/disclaimer.txt").read_text())


# Entry point for the script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    main()
