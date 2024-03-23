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
