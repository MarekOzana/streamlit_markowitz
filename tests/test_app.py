"""
Unit tests on the main streamlit app

.. author:: Marek Ozana
.. date:: 2024-03
"""

from streamlit.testing.v1 import AppTest


def test_app():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert not at.exception


def test_app_with_adjusted_r_min():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Initial check to ensure the app started correctly.
    assert not at.exception, "The app should start without exceptions."
    assert at.markdown[0].value == "Optimal Portfolio: r=7.0%, vol=2.6%"

    # Set the r_min slider to 8%.
    r_min_slider = at.sidebar.slider[0]
    r_min_slider.set_value(8.0).run()  # Set to 8%

    # Verify that the app didn't throw an exception after the change.
    assert not at.exception, "The app should not throw after setting r_min to 8%."
    assert at.markdown[0].value == "Optimal Portfolio: r=8.0%, vol=3.9%"


def test_app_remove_tickers():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Initial check to ensure the app started correctly.
    assert not at.exception, "The app should start without exceptions."
    assert at.markdown[0].value == "Optimal Portfolio: r=7.0%, vol=2.6%"

    at.sidebar.multiselect[0].unselect("Climate Focus").run()

    # Verify that the app didn't throw an exception after the change.
    assert not at.exception, "The app should not throw removing ticker"
    assert at.markdown[0].value == "Optimal Portfolio: r=7.0%, vol=2.9%"
