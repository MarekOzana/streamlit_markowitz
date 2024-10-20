"""
Unit tests on the main streamlit app

.. author:: Marek Ozana
.. date:: 2024-03
"""

from streamlit.testing.v1 import AppTest
import logging


def test_app():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert not at.exception


def test_app_with_adjusted_r_min():
    at = AppTest.from_file("scripts/Markowitz.py", default_timeout=30)
    at.run()

    # Initial check to ensure the app started correctly.
    assert not at.exception, "The app should start without exceptions."
    assert "**Expected Return in 1y** = 6.0%" in at.markdown[0].value
    assert "**Expected volatility** = 2.4%" in at.markdown[0].value

    # Set the r_min slider to 8%.
    r_min_slider = at.sidebar.slider[0]
    r_min_slider.set_value(8.0).run()  # Set to 8%

    # Verify that the app didn't throw an exception after the change.
    assert not at.exception, "The app should not throw after setting r_min to 8%."
    assert "**Expected Return in 1y** = 7.5%" in at.markdown[0].value
    assert "**Expected volatility** = 11.1%" in at.markdown[0].value


def test_app_remove_tickers():
    at = AppTest.from_file("scripts/Markowitz.py", default_timeout=30)
    at.run()

    # Initial check to ensure the app started correctly.
    assert not at.exception, "The app should start without exceptions."
    assert "**Expected Return in 1y** = 6.0%" in at.markdown[0].value
    assert "**Expected volatility** = 2.4%" in at.markdown[0].value

    at.sidebar.multiselect[0].unselect("Climate Focus").run()

    # Verify that the app didn't throw an exception after the change.
    assert not at.exception, "The app should not throw removing ticker"
    assert "**Expected Return in 1y** = 6.0%" in at.markdown[0].value
    assert "**Expected volatility** = 2.6%" in at.markdown[0].value


def test_MicroFinanceAnalyzer_smoke_test(caplog):
    at = AppTest.from_file("scripts/Micro_Finance_Analyzer.py", default_timeout=30)

    with caplog.at_level(logging.DEBUG):
        at.run()
    assert not at.exception
    assert "Reading in data and calculating params" in caplog.text
    assert "Micro Finance Analyzer" in at.title[0].value


def test_MicroFinanceAnalyzer_page_switch():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    at.switch_page(page_path="scripts/Micro_Finance_Analyzer.py")
    at.run()
    assert not at.exception


def test_HistoricalRiskReturn_smoke_test(caplog):
    at = AppTest.from_file("scripts/Historical_Risk_Return.py", default_timeout=30)

    with caplog.at_level(logging.DEBUG):
        at.run()
    assert not at.exception
    assert "Historical Risk Return Analysis" in at.title[0].value


def test_HistoricalRiskReturn_page_switch():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    at = at.switch_page(page_path="scripts/Historical_Risk_Return.py")
    at.run()
    assert not at.exception
