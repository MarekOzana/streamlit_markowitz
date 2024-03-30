"""
Markowitz Optimization on Funds
author: 2024-03 Marek Ozana
"""

import pathlib

import polars as pl
import polars.selectors as cs

import streamlit as st
import logging
from src.data_manager import DataManager
import src.charts as charts
import src.optimization as opt

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
    col2.altair_chart(f_w.properties(title=title), use_container_width=True)
    st.markdown(title)

    # Cumulative returns chart
    r_cum = db.get_cumulative_rets_with_OPT(db.ret_vol["name"].to_list(), w)
    fig = charts.create_cum_ret_chart(r_cum)
    st.altair_chart(fig, use_container_width=True)

    # Create chart with expected returns & Prob of negative returns
    r_fig, p_fig = charts.create_exp_ret_chart(r_ann=r_opt, vol_ann=vol_opt, n=36)
    st.altair_chart(r_fig, use_container_width=True)
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
        col1, col2 = st.columns([4, 1])
        fund_name = col2.selectbox(
            label="Fund", options=db.names(), label_visibility="collapsed"
        )
        df = db.get_cumulative_rets_and_dd(name=fund_name)
        fig = charts.create_cumul_ret_with_drawdown_chart(df)
        col1.altair_chart(fig, use_container_width=True)

    st.divider()
    st.caption(pathlib.Path("data/disclaimer.txt").read_text())


# Entry point for the script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    main()
