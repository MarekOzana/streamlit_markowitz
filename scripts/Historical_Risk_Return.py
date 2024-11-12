"""
Historical RIsk Return Analysis
(c) 2024-10 Marek Ozana
"""

import logging
import math
from datetime import date
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

st.set_page_config(
    page_title="Historical Risk Return Analysis",
    layout="wide",
)

logger: logging.Logger = logging.getLogger(__name__)


def get_monthly_rets(tickers: list[str], start_dt: date) -> pl.DataFrame:
    """Get monthly returns for given tickers since start_dt

    Returns
    -------
    pl.DataFrame: Schema([('security', String),
                          ('date', Date),
                          ('r', Float64)])
    """
    # Read the data from the parquet file
    df = pl.read_parquet("data/ix_data.parquet")

    # Filter the data for the given tickers and start date
    filtered_df = df.filter(
        pl.col("security").is_in(tickers) & (pl.col("date") >= start_dt)
    )

    # Sort the filtered data by security and date
    sorted_df = filtered_df.sort(by=["security", "date"])

    # Calculate daily returns and fill null values with 0
    daily_rets_df = sorted_df.with_columns(
        pl.col("value").pct_change().over("security").fill_null(0).alias("r")
    )

    # Aggregate daily returns to monthly returns
    monthly_rets_df = daily_rets_df.group_by_dynamic(
        "date", every="1mo", group_by="security"
    ).agg(
        pl.col("date").last().alias("dt_end"),
        pl.col("r").add(1).product().add(-1),
    )

    # Find the first date where all securities have data available
    first_dt: date = (
        monthly_rets_df.group_by("security")
        .agg(pl.col("date").min())
        .select(pl.col("date").max())
        .item()
    )

    # Filter the monthly returns to start from the first common date
    result_df: pl.DataFrame = monthly_rets_df.filter(pl.col("date") >= first_dt).select(
        pl.col("dt_end").alias("date"), "security", "r"
    )

    return result_df


def calc_portf_rets(
    r_m: pl.DataFrame, weights: dict[str, float], name: str = "PORTF"
) -> pl.DataFrame:
    """Calculate portfolio returns. Based on monthly data.
    with fixed monthly rebalancing.

    Params
    ------
    r_m (pl.DataFrame): monthly return table with cols (date, dt_end, security, r)
    weights (dict): mapping between security and portfolio weight
    name (str): portfolio name. Used as security name

    Returns
    -------
    rets (pl.DataFrame): portfolio return table with cols (date, security, r)
    """
    rets: pl.DataFrame = (
        r_m.filter(pl.col("security").is_in(weights.keys()))
        .with_columns(
            pl.col("security")
            .map_elements(
                lambda x: weights.get(x), returns_scalar=True, return_dtype=pl.Float64
            )
            .alias("w")
        )
        .group_by(["date"])
        .agg((pl.col("r") * pl.col("w")).sum().alias("r"))
        .sort(by=pl.col("date"))
    ).select("date", pl.lit(name).alias("security"), "r")
    return rets


def calc_rets(r_m: pl.DataFrame, scale: int) -> pl.DataFrame:
    """Calculate annualized returns for each sec in 'security' column,

    Params
    ------
    r_m (pl.DataFrame): A DataFrame containing the returns data with columns 'security', 'date', and 'r'.
    scale (int): number of rows per year. 12 for monthly, 252 for daily returns

    Rets
    ----
    pl.DataFrame: (security, r) where 'r' is annualized return
    """
    return r_m.group_by(pl.col("security")).agg(
        pl.col("r").add(1).product().pow(scale / pl.col("r").count()).add(-1),
    )


def calc_vols(r_m: pl.DataFrame, scale: int) -> pl.DataFrame:
    """Calculate annualized volatilities.

    Corresponds to Excel STDEV.S

    Params
    ------
    r_m (pl.DataFrame): A DataFrame containing the returns data with columns 'security', 'date', and 'r'.
    scale (int): number of rows per year. 12 for monthly, 252 for daily returns

    Rets
    ----
    pl.DataFrame: (security, vol) where 'vol' is annualized volatility
    """
    return r_m.group_by(pl.col("security")).agg(
        pl.col("r").std().mul(math.sqrt(scale)).alias("vol")
    )


def calc_portfolio_metrics(r_m: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    """Calculate list of portfolios with weights, returns, and volatilities.

    Params
    ------
    r_m (pl.DataFrame): A DataFrame containing the returns data with columns 'security', 'date', and 'r'.
    tickers (list[str]): List of tickers for which to calculate portfolio metrics.

    Returns
    -------
    pl.DataFrame: A DataFrame containing the portfolio metrics with the following schema:
        - name (pl.String): Portfolio name in the format "w0%/w1%/w2%"
        - r (pl.Float64): Annualized return
        - vol (pl.Float64): Annualized volatility
        - r2vol (pl.Float64): Return to volatility ratio
        - w0 (pl.Float64): Weight of the first ticker
        - w1 (pl.Float64): Weight of the second ticker
        - w2 (pl.Float64): Weight of the third ticker
    """
    assert len(tickers) == 3, "Only 3 tickers supported"

    def _gen_weights():
        """Generate all possible weight combinations for three assets."""
        for w0 in [i * 0.1 for i in range(11)]:
            for w1 in [i * 0.1 for i in range(11)]:
                if w0 + w1 <= 1:
                    w2 = round(1 - w0 - w1, 2)
                    yield w0, w1, w2

    def _calc_metrics(r_m, weights):
        """Calculate portfolio metrics for given weights."""
        weights_dict = dict(zip(tickers, weights))
        p_name = f"{weights[0]:0.0%}/{weights[1]:0.0%}/{weights[2]:0.0%}"
        r_p = calc_portf_rets(r_m, weights=weights_dict, name=p_name)
        r_ann = calc_rets(r_p, scale=12)["r"].item(0)
        vol_ann = calc_vols(r_p, scale=12)["vol"].item(0)
        r2vol = r_ann / (vol_ann + 1e-9)
        return p_name, r_ann, vol_ann, r2vol, *weights

    lst_metrics = [_calc_metrics(r_m, weights) for weights in _gen_weights()]

    portfolios = pl.DataFrame(
        lst_metrics,
        schema=[
            ("name", pl.String),
            ("r", pl.Float64),
            ("vol", pl.Float64),
            ("r2vol", pl.Float64),
            ("w0", pl.Float64),
            ("w1", pl.Float64),
            ("w2", pl.Float64),
        ],
        orient="row",
    ).with_columns(pl.exclude("name").round(4))
    return portfolios


def chart_risk_return(
    portfolios: pl.DataFrame, tickers: list[str], title: str
) -> alt.LayerChart:
    """Create a risk-return chart for given portfolio metrics.

    We assume that w0 = GOVIES, w1=HY /AT1, w2=IG
    """
    assert len(tickers) == 3, "Only 3 tickers supported"

    g_data = portfolios.with_columns((pl.col("r") / pl.col("vol")).alias("r2vol"))

    base = (
        alt.Chart(
            g_data,
            title=alt.Title(
                text=title,
                subtitle=f"{tickers[0]} / {tickers[1]} / {tickers[2]}",
            ),
        )
        .mark_point()
        .encode(
            x=alt.X("vol:Q")
            .axis(format="%")
            .scale(zero=False)
            .title("Annualized Volatility [%]"),
            y=alt.Y("r:Q")
            .axis(format="%")
            .scale(zero=False)
            .title("Annualized Return [%]"),
            tooltip=[
                alt.Tooltip("r", format="0.2%"),
                alt.Tooltip("vol:Q", format="0.2%"),
                alt.Tooltip("w0", format="0.0%").title(tickers[0]),
                alt.Tooltip("w1", format="0.0%").title(tickers[1]),
                alt.Tooltip("w2", format="0.0%").title(tickers[2]),
            ],
        )
    )

    def _create_layer(filter_condition, c_scheme):
        return (
            base.mark_point(filled=True)
            .transform_filter(filter_condition)
            .encode(
                color=alt.Color("r2vol:Q").scale(scheme=c_scheme).legend(None),
                text=alt.Text("name:N"),
            )
        )

    scatter = base.transform_filter(
        (alt.datum.w0 > 0.0) & (alt.datum.w1 > 0) & (alt.datum.w2 > 0)
    ).encode(color=alt.Color("r2vol:Q", scale=alt.Scale(scheme="greys"), legend=None))

    gov_hy = _create_layer(alt.datum.w2 < 0.001, "greens")
    gov_hy_txt = gov_hy.mark_text(align="right", baseline="bottom")
    gov_ig = _create_layer(alt.datum.w1 < 0.001, "reds")
    gov_ig_txt = gov_ig.transform_filter((alt.datum.w0 * 100 % 30) == 0).mark_text(
        align="left", baseline="top"
    )
    ig_hy = _create_layer(alt.datum.w0 < 0.001, "blues")
    ig_hy_txt = ig_hy.transform_filter((alt.datum.w2 * 100 % 30) == 0).mark_text(
        align="left", baseline="middle", dx=5
    )

    layers = [scatter, gov_hy, gov_hy_txt, gov_ig, gov_ig_txt, ig_hy, ig_hy_txt]

    return alt.layer(*layers).resolve_scale(color="independent")


def chart_portf_cumul_rets(
    r_m: pl.DataFrame, weights: dict[str, float]
) -> alt.LayerChart:
    """Chart cumulative returns for Govt, IG, HY and portfolio defined by weights"""
    p_name = ""
    for key, value in weights.items():
        if value > 0:
            if len(p_name) > 0:
                p_name += " / "
            p_name += f"{key} {value:.0%}"
    r_p = calc_portf_rets(r_m, weights=weights, name=p_name)
    g_data = (
        pl.concat([r_m, r_p])
        .sort(by=["security", "date"])
        .select(
            pl.col("date"),
            pl.col("security"),
            pl.col("r").add(1).cum_prod().over("security").add(-1).alias("r_cum"),
        )
    )
    # add zero to previous month
    df_zero = (
        r_m.group_by("security")
        .agg((pl.col("date").min() - pl.duration(days=22)).alias("date"))
        .select("date", "security", pl.lit(0.0).alias("r_cum"))
    )
    g_data = pl.concat([df_zero, g_data])

    # Create chart
    base = alt.Chart(
        g_data, title=f"Total Returns since {g_data['date'].min()}"
    ).encode(
        x=alt.X("date:T").title(None),
        y=alt.Y("r_cum:Q").title("Cumulative Return [%]").axis(format="%"),
        color=alt.Color("security:N").title("Portfolio").legend(orient="top-left"),
    )
    lines = base.mark_line().encode(
        size=alt.condition(alt.datum.security == p_name, alt.value(4), alt.value(2)),
    )
    txt = base.mark_text(align="left", baseline="middle").encode(
        x=alt.X("date:T").aggregate("max"),
        y=alt.Y("r_cum:Q").aggregate(argmax="date"),
        text=alt.Text("r_cum:Q", format="0.0%").aggregate(argmax="date"),
    )
    return lines + txt


def get_user_input():
    with st.sidebar:
        st.title("Parameters")

        start_year = st.slider(
            "Start Year",
            min_value=1999,
            max_value=2023,
            value=2004,
        )
        start_dt = date(start_year, 1, 1)
        tickers = st.selectbox(
            "Select tickers",
            options=[
                ("EU Govt 3-5y", "EU HY", "EU IG"),
                ("US Trsy 3-5y", "US HY", "US IG"),
                ("US Trsy 3-5y", "US HY", "S&P 500"),
                ("EU Govt 3-5y", "EU HY", "EuroStoxx 50"),
                ("EU Govt 3-5y", "CoCo H-EUR", "EU HY"),
                ("Sweden Govt", "SEB Hybrid G-SEK", "Sweden Eq"),
                ("EU Govt 3-5y", "SEB Hybrid R-EUR", "EuroStoxx 50")
            ],
        )
        height = st.slider(
            "Chart Height", min_value=200, max_value=800, value=400, step=50
        )
        r_m: pl.DataFrame = get_monthly_rets(tickers=tickers, start_dt=start_dt)

    return tickers, r_m, height


def main() -> None:
    st.title("Historical Risk Return Analysis")
    tickers, r_m, height = get_user_input()

    with st.spinner("Calculating Portfolio Metrics..."):
        # Risk Return Charts
        portfolios = calc_portfolio_metrics(r_m, tickers)
        title = f"Realized Return & Volatility ({r_m['date'].min():%b%Y} - {r_m['date'].max():%b%Y})"
        fig_pf = chart_risk_return(portfolios, tickers, title=title).properties(
            height=height
        )
        st.altair_chart(fig_pf, use_container_width=True)

    # Cumulative Returns Charts
    st.markdown("## Total Returns Chart")
    portfolios = portfolios.sort(by="r2vol", descending=True)
    c1, c2 = st.columns([0.4, 1])
    with c1:  # Portfolio Metrics Table
        df = portfolios.with_columns(pl.exclude(["name", "r2vol"]).mul(100))
        portf = st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Portfolio"),
                "r": st.column_config.NumberColumn(
                    "Return", help="Annualized Return", format="%0.1f %%"
                ),
                "vol": st.column_config.NumberColumn(
                    "Volatility", help="Annualized Volatility", format="%0.1f %%"
                ),
                "r2vol": st.column_config.NumberColumn(
                    "Ret/Vol", help="Volatility Adjusted Return", format="%0.2f"
                ),
                "w0": st.column_config.NumberColumn(tickers[0], format="%0.0f %%"),
                "w1": st.column_config.NumberColumn(tickers[1], format="%0.0f %%"),
                "w2": st.column_config.NumberColumn(tickers[2], format="%0.0f %%"),
            },
            selection_mode="single-row",
            on_select="rerun",
        )
        st.session_state["sel_row"] = portf.selection.rows
    with c2:
        weights = {tickers[0]: 0.5, tickers[1]: 0.5}  # default values
        if "sel_row" not in st.session_state:
            st.session_state["sel_row"] = [0]
        if len(st.session_state["sel_row"]) > 0:
            row = portfolios.row(st.session_state["sel_row"][0], named=True)
            weights = {tickers[i]: row[f"w{i}"] for i in range(3)}

        fig_rets = chart_portf_cumul_rets(r_m, weights).properties(height=height)
        st.altair_chart(fig_rets, use_container_width=True)
    st.divider()
    st.caption(Path("data/disclaimer.txt").read_text())


# Entry point for the script
if __name__ == "__main__" or __name__ == "__page__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    main()
