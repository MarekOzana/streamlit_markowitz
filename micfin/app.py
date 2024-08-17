"""
Analyze Micro Finance in portfolio

author: 2024-08 Marek Ozana
"""

import logging
import streamlit as st
import pandas as pd

# import sys
# import os

# # Add the src directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
# import optimization as opt

logger: logging.Logger = logging.getLogger(__name__)


def get_user_input() -> None:
    with st.sidebar:
        st.title("Parameters")
        r_min = (
            st.slider(
                "Min Required Return",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                value=7.0,
                format="%.1f%%",
                help="Constraint on minimum required return",
            )
            / 100
        )
        st.session_state["r_min"] = r_min

        with st.popover("Exp Returns"):
            exp_rets = st.data_editor(
                st.session_state["orig_exp_rets"] * 100,
                use_container_width=True,
                column_config={
                    "exp_ret": st.column_config.NumberColumn(
                        "Exp Return [%]", format="%0.1f %%"
                    ),
                },
            )
            st.session_state["exp_rets"] = exp_rets.div(100)

        with st.popover("Volatilities"):
            vols = st.data_editor(
                st.session_state["orig_vols"] * 100,
                use_container_width=False,
                column_config={
                    "Vol": st.column_config.NumberColumn(
                        "Volatility [%]", format="%0.1f %%"
                    ),
                },
            )
            st.session_state["vols"] = vols.div(100)

        with st.popover("Correlations"):
            corr = st.data_editor(
                st.session_state["orig_corr"] * 100,
                use_container_width=True,
                column_config={
                    col: st.column_config.NumberColumn(
                        format="%.0f%%", min_value=-100, max_value=100
                    )
                    for col in st.session_state["orig_corr"].columns
                },
            )
            st.session_state["corr"] = corr.div(100)


def load_data() -> None:
    if "corr" in st.session_state:
        logger.info("ignoring load_data because data already exist")
        return

    logger.info("Reading in data and calculating params")
    rets = pd.read_csv("data/micfin_q_rets.csv", index_col=0, parse_dates=True)
    st.session_state["orig_rets"] = rets

    exp_rets = pd.read_csv("data/micfin_exp_rets.csv", index_col=0).squeeze()
    st.session_state["orig_exp_rets"] = exp_rets

    vols = rets.std() * 2  # * sqrt(4) for quarterly
    vols.name = "Vol"
    st.session_state["orig_vols"] = vols

    corr = rets.corr()
    st.session_state["orig_corr"] = corr.round(2)


def main() -> None:
    st.title("Micro Finance Analyzer")
    load_data()
    get_user_input()
    st.dataframe(st.session_state["corr"])
    st.dataframe(st.session_state["exp_rets"])
    st.dataframe(st.session_state["vols"])


# Entry point for the script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    main()
