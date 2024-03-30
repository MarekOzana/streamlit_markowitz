"""
Charts for Markowitz Optimization Visualization

.. author:: Marek Ozana
.. date:: 2024-03
"""

import altair as alt
import polars as pl
import numpy as np
from scipy.stats import norm
from datetime import date


def create_scatter_chart(g_data: pl.DataFrame) -> alt.LayerChart:
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
        base.transform_fold(["Expected", "Worst (95%)"], as_=["Type of Return", "ret"])
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


def create_cum_ret_chart(r_cum: pl.DataFrame) -> alt.Chart:
    """Create chart with cumulative returns
    based on long format dataframe
    """
    r_cum = r_cum.with_columns(
        (pl.col("date") == pl.col("date").max()).alias("is_last")
    )
    order = {"SEB Hybrid": 0, "OPTIMAL": 1}
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
            alt.FieldOneOfPredicate(field="name", oneOf=["SEB Hybrid", "OPTIMAL"]),
            alt.value(1),
            alt.value(0.6),
        ),
        size=alt.condition(
            alt.FieldOneOfPredicate(field="name", oneOf=["SEB Hybrid", "OPTIMAL"]),
            alt.value(3),
            alt.value(1.25),
        ),
    )
    text = (
        base.mark_text(align="left", baseline="middle")
        .encode(text=alt.Text("return:Q").format("0.0%"))
        .transform_filter(alt.datum.is_last)
    )
    fig = line + text
    return fig


def create_cumul_ret_with_drawdown_chart(df: pl.DataFrame) -> alt.LayerChart:
    """Create cumulative return for single time series including drwadowns

    Parameters
    ----------
    df: pl.DataFrame
        expected columns ["date", name, "DrawDown]
    """
    name: str = df.columns[1]
    base = alt.Chart(df.to_pandas()).encode(
        x=alt.X("date:T").title(None),
        tooltip=[
            "date:T",
            alt.Tooltip(name, format="0.2%"),
            alt.Tooltip("DrawDown:Q", format="0.2%"),
        ],
    )
    f_ret = base.mark_line().encode(y=alt.Y(f"{name}:Q").axis(format="%").title(name))
    f_dd = base.mark_area(
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="firebrick", offset=0),
                alt.GradientStop(color="white", offset=1),
            ],
            x1=1,
            x2=1,
            y1=1,
            y2=0,
        ),
    ).encode(y=alt.Y("DrawDown").axis(format="%").title(""))
    fig = f_dd + f_ret
    return fig
