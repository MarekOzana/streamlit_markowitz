"""
Markowitz Optimization on Funds
author: 2024-03 Marek Ozana
"""

from datetime import date


import altair as alt

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy.optimize as sco
from scipy.stats import norm
import streamlit as st
import logging
from src.data_manager import DataManager

st.set_page_config(page_title="Markowitz", layout="wide", page_icon="data/app_icon.png")

logger: logging.Logger = logging.getLogger(__name__)



class ChartManager:
    @staticmethod
    def create_scatter_chart(g_data: pl.DataFrame) -> alt.Chart:
        base = alt.Chart(g_data, title="Risk vs Return Profile").encode(
            x=alt.X("vols").axis(format="%").title("Volatility"),
            y=alt.Y("rets").axis(format="%").title("Expected Return"),
            tooltip=[
                "name",
                alt.Tooltip("w_opt:Q", format="0.1%", title="Optimal Weight"),
                alt.Tooltip("rets:Q", format="0.1%", title="Expected Return"),
                alt.Tooltip("vols", format="0.1%", title="Expected Volatility"),
            ],
        )
        sc = base.mark_point(filled=True, stroke="grey").encode(
            color=alt.Color("w_opt:Q")
            .scale(domain=[0, 1], scheme="blueorange")
            .legend(None),
            size=alt.Size("w_opt:Q").scale(range=[15, 350]).legend(None),
            shape=alt.condition(
                alt.datum.name == "OPTIMAL", alt.value("diamond"), alt.value("circle")
            ),
        )
        text = base.mark_text(align="center", baseline="bottom").encode(
            text=alt.Text("name")
        )
        fig = sc + text
        return fig

    @staticmethod
    def create_portf_weights_chart(g_data: pl.DataFrame):
        base = (
            alt.Chart(g_data, title="Optimal Portfolio").encode(
                x=alt.X("name:N").title(None),
                y=alt.Y("w_opt")
                .title("Optimal Weight")
                .axis(format="%")
                .scale(domain=[0, 1], scheme="blueorange"),
                tooltip=[
                    "name",
                    alt.Tooltip("w_opt:Q", format="0.1%", title="Optimal Weight"),
                    alt.Tooltip("rets:Q", format="0.1%", title="Expected Return"),
                    alt.Tooltip("vols:Q", format="0.1%", title="Expected Volatility"),
                ],
            )
        ).transform_filter(alt.datum.name != "OPTIMAL")
        bars = base.mark_bar(stroke="grey").encode(
            color=alt.Color("w_opt:Q")
            .scale(domain=[0, 1], scheme="purpleorange")
            .legend(None)
        )
        text = base.mark_text(align="center", baseline="bottom").encode(
            text=alt.Text("w_opt", format="0.0%")
        )
        fig = bars + text
        return fig

    @staticmethod
    def create_exp_ret_chart(
        r_ann: float = 0.05,
        vol_ann: float = 0.08,
        n: int = 12,
    ) -> tuple:
        """Plot Expected returns and Probability of negative returns

        Parameters
        ==========
            r_ann: annualized expected return
            vol_ann: annualized volatility
            n: number of months in chart

        Returns
        =======
            tuple(alt.Chart, alt.Chart)
        """
        # Prepare DataFrame with dates, time (in years), exp returns and probabilities
        month = np.arange(0, n + 1, step=1)
        t = month.astype(np.float64) / 12.0
        t[0] = 1e-15
        today = date.today()
        n_std: float = norm.ppf(0.95)  # number of standard deviations for 95%
        g_data: pl.DataFrame = pl.DataFrame(
            {
                "month": month,
                "date": pl.date_range(
                    today,
                    date(today.year + n // 12, (today.month + n + 1) % 12, 1),
                    "1mo",
                    eager=True,
                ),
                # Lowest expected return with 95% probability
                "Worst (95%)": np.exp(r_ann * t - n_std * vol_ann * np.sqrt(t)) - 1,
                # Expected return
                "Expected": np.exp(r_ann * t) - 1,
                # Probability of negative return
                "p_neg": norm.cdf(0, loc=r_ann * t, scale=vol_ann * np.sqrt(t)),
            }
        )

        # Create Chart
        base = alt.Chart(g_data.to_pandas()).encode(
            x=alt.X("yearmonth(date):T").title(None)
        )
        r_fig = (
            base.transform_fold(
                ["Expected", "Worst (95%)"], as_=["Type of Return", "ret"]
            )
            .mark_line()
            .encode(
                y=alt.Y("ret:Q").title("Return [%]").axis(format="%"),
                color=alt.Color("Type of Return:N"),
                tooltip=[
                    "date:T",
                    "Type of Return:N",
                    alt.Tooltip("ret:Q").format("0.2%"),
                ],
            )
            .properties(title="Expected and Worst (95%) Returns")
        )

        p_fig = (
            base.mark_bar()
            .encode(
                y=alt.Y("p_neg:Q").axis(format="%").title("Negative Ret Prob"),
                color=alt.Color("p_neg:Q").legend(None).scale(scheme="blueorange"),
                tooltip=[
                    "date:T",
                    alt.Tooltip("p_neg:Q").format("0.2%").title("Probability"),
                ],
            )
            .properties(title="Probability of Negative Return")
        )
        return r_fig, p_fig

    @staticmethod
    def create_cum_ret_chart(r_cum: pl.DataFrame) -> alt.Chart:
        """Create chart with cumulative returns
        based on long format dataframe
        """
        r_cum = r_cum.with_columns(
            (pl.col("date") == pl.col("date").max()).alias("is_last")
        )
        order = {"Hybrid": 0, "OPTIMAL": 1}
        ord_tickers = sorted(
            r_cum["name"].unique().to_list(), key=lambda x: order.get(x, float("inf"))
        )
        base = alt.Chart(r_cum.to_pandas(), title="Cumulative Returns").encode(
            x=alt.X("date:T").title(None),
            y=alt.Y("return:Q").title("Cumulative Return").axis(format="%"),
            color=alt.Color("name:N").sort(ord_tickers).scale(scheme="category10"),
            tooltip=["name:N", "date:T", alt.Tooltip("return:Q", format="0.2%")],
        )
        line = base.mark_line().encode(
            opacity=alt.condition(
                alt.FieldOneOfPredicate(field="name", oneOf=["Hybrid", "OPTIMAL"]),
                alt.value(1),
                alt.value(0.6),
            ),
            size=alt.condition(
                alt.FieldOneOfPredicate(field="name", oneOf=["Hybrid", "OPTIMAL"]),
                alt.value(3),
                alt.value(1.25),
            ),
        )
        text = (
            base.mark_text(align="left", baseline="middle")
            .encode(text=alt.Text("return:Q").format("0.0%"))
            .transform_filter((alt.datum.is_last == True))
        )
        fig = line + text
        return fig


class OptimizationManager:
    @staticmethod
    def find_min_var_portfolio(
        exp_rets: np.array, cov: np.array, r_min: float = 0, w_max: float = 1
    ):
        """Find portfolio with minimum variance given constraint return
        Solve the following optimization problem
            min: w.T*COV*w
            subjto: w.T * r_ann >= r_min
                    w.T * w = 1
                    0 <= w[i] <= w_max for every i
        Parameters
        ==========
            exp_rets: annualized expected returns
            cov: covariance matrix
            r_min: minimum portfolio return (constraint)
            w_max: maximum individual weight (constraint)
        Returns
        =======
            (w, r_opt, vol_opt)
            w: portfolio weights
            r_opt: return of optimal portfolio
            vol_opt: volatility of optimal portfolio
        """

        def calc_var(w, cov):
            """Calculate portfolio Variance"""
            return np.dot(w.T, np.dot(cov, w))

        n_assets = len(exp_rets)
        constraints = [
            # sum(w_i) = 1
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            # sum(r_i * w_i >= r_min)
            {"type": "ineq", "fun": lambda x: np.dot(x.T, exp_rets) - r_min},
        ]
        bounds = tuple((0, w_max) for asset in range(n_assets))  # sequence of (min,max)

        opts = sco.minimize(
            # Objective Function
            fun=calc_var,
            # Initial guess
            x0=n_assets * [1.0 / n_assets],
            # Extra Arguments to objective function
            args=(cov,),
            method="SLSQP",
            options={"ftol": 1e-7, "maxiter": 100},
            bounds=bounds,
            constraints=constraints,
        )
        w = opts["x"]
        r_opt = np.dot(w, exp_rets)
        vol_opt = np.sqrt(calc_var(w, cov))
        return w, r_opt, vol_opt


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
    # Set tickers
    if ("tickers" not in st.session_state) or (tickers != st.session_state["tickers"]):
        st.session_state["tickers"] = tickers
        db.set_ret_vol_corr(tickers)
    return r_min


def create_main_tab(db: DataManager, r_min: float) -> None:
    """Create main TAB with optimization charts and portfolio Weights"""
    # Optimize
    w, r_opt, vol_opt = OptimizationManager.find_min_var_portfolio(
        db.get_ret(), db.get_covar(), r_min
    )

    # Create Scatter chart and Portfolio Composition Chart
    g_data = pl.DataFrame(
        {
            "name": db.ret_vol["name"].to_list() + ["OPTIMAL"],
            "vols": db.get_vol().tolist() + [vol_opt],
            "rets": db.get_ret().tolist() + [r_opt],
            "w_opt": w.tolist() + [1],
        }
    )
    f_sc = ChartManager.create_scatter_chart(g_data)
    f_w = ChartManager.create_portf_weights_chart(g_data)
    title = f"Optimal Portf: r={r_opt:0.1%}, vol={vol_opt:0.1%}"
    col1, col2 = st.columns([1.5, 1])
    col1.altair_chart(f_sc, use_container_width=True)
    col2.altair_chart(f_w.properties(title=title), use_container_width=True)

    # Cumulative returns chart
    r_cum = db.get_cumulative_rets_with_OPT(db.ret_vol["name"].to_list(), w)
    fig = ChartManager.create_cum_ret_chart(r_cum)
    st.altair_chart(fig, use_container_width=True)

    # Create chart with expected returns & Prob of negative returns
    r_fig, p_fig = ChartManager.create_exp_ret_chart(r_ann=r_opt, vol_ann=vol_opt, n=36)
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
        with st.spinner("Updating from Yahoo"):
            db.update_from_yahoo()


USAGE: str = """
Welcome to the Optimal Portfolio Generator! This app aids in creating an optimal 
portfolio based on your required minimum return.

**How to Use the App:**
1. **Set Your Required Return:** Adjust the slider in the sidebar to set your 
   desired minimum return. 
2. **Select Tickers:** Pick stocks or assets from the multiselect dropdown. Your 
   portfolio will be built based on these choices.
3. **Analyze the Portfolio:** With your return set and tickers chosen, the app 
   displays the optimal portfolio. It includes a risk vs. return scatter chart, 
   asset weight bar chart, and other charts showing historical and expected 
   returns.
4. **Iterate:** Modify the return or tickers to explore how the portfolio adjusts. 
   This helps you make informed decisions by understanding the trade-offs.
5. **Adjust Assumptions:** On the *DATA* tab, tweak expected returns, volatilities, 
   and correlations. Initial values are based on daily returns since the start of 2023.

Use this tool to explore how varying parameters influence your portfolio's risk 
and return profile. Happy investing!
"""


DISCLAIMER: str = """
**Disclaimer:**  
The Optimal Portfolio Generator is for informational purposes only and is not intended  
as financial advice, nor does it constitute an offer, invitation, or recommendation  
to buy or sell securities. We do not guarantee the accuracy or completeness of  
the information provided.

Investors are encouraged to seek independent legal, financial, and tax advice before  
making investment decisions. The material has not been reviewed by regulatory  
authorities and is not intended for distribution in jurisdictions where its distribution  
is prohibited.

Investing involves risks, including the potential loss of principal. Past performance  
is not indicative of future results. The value of investments can fluctuate, and  
investors may not get back the amount invested. Derivatives involve high risks and  
are not suitable for all investors.

This material does not account for individual circumstances and should not be  
considered tailored investment advice. Investors should consult their advisors  
and review the fund's objectives, risks, charges, and expenses before investing.  
Read the prospectus and Key Information Documents available for more details.

We assume no liability for any loss or damage arising from the use of this app.  
Portfolio composition may change, and past information may not reflect current  
portfolio characteristics. Currency exchange rates can affect returns, and no  
hedging strategy guarantees performance.

This information is current as of the date indicated and is subject to change without  
notice. Recipients are responsible for their investment decisions, and this material  
should not be viewed as investment advice or a recommendation.

For investment advice tailored to your circumstances, please contact your investment  
advisor. You are solely responsible for your investment decisions.
"""


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
        st.divider()
        with st.popover("How to Use the App"):
            st.markdown(USAGE)

    tab_main, tab_data = st.tabs(["Optimal Portfolio", "Data"])
    with tab_main:
        create_main_tab(db, r_min)

    with tab_data:
        create_edit_assumptions_tab(db)

    st.divider()
    st.caption(DISCLAIMER)


# Entry point for the script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    main()
