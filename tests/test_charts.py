"""
Unit test on charts module
"""

import polars as pl
import altair as alt
import src.charts


def test_create_scatter_chart():
    # Mock data to test the function
    g_data = pl.DataFrame(
        {
            "name": ["A", "B"],
            "vols": [0.1, 0.2],
            "rets": [0.1, 0.2],
            "w_opt": [0.5, 0.5],
        }
    )

    # Generate the chart
    chart = src.charts.create_scatter_chart(g_data)

    # Check if the output is an Altair Chart
    assert isinstance(chart, alt.LayerChart)

    # Check the chart title
    assert chart.layer[0].title == "Risk vs Return Profile"

    # Check if x and y encodings are correctly set
    assert "vols" in chart.layer[0].encoding.x.shorthand
    assert "rets" in chart.layer[0].encoding.y.shorthand


def test_create_portf_weights_chart():
    g_data = pl.DataFrame(
        {
            "name": ["A", "B"],
            "vols": [0.1, 0.2],
            "rets": [0.1, 0.2],
            "w_opt": [0.5, 0.5],
        }
    )
    chart = src.charts.create_portf_weights_chart(g_data)
    # Generate the chart
    chart = src.charts.create_scatter_chart(g_data)

    # Check if the output is an Altair Chart
    assert isinstance(chart, alt.LayerChart)

    # Check the chart title
    assert chart.layer[0].title == "Risk vs Return Profile"

    # Check if x and y encodings are correctly set
    assert "vols" in chart.layer[0].encoding.x.shorthand
    assert "rets" in chart.layer[0].encoding.y.shorthand


def test_create_exp_ret_chart():
    r_fig, p_fig = src.charts.create_exp_ret_chart(r_ann=0.05, vol_ann=0.08, n=12)
    assert isinstance(r_fig, alt.Chart)
    assert isinstance(p_fig, alt.Chart)
