"""
Unit tests on data_manager module

.. author:: Marek Ozana
.. date:: 2024-03
"""

import pytest
from src.data_manager import DataManager
from pathlib import Path


# Fixture to create a DataManager instance
@pytest.fixture
def dm():
    dm = DataManager(
        fund_tbl=Path("tests/data/t_fund.csv"),
        price_tbl=Path("tests/data/t_price.parquet"),
    )
    return dm


def test__init__(dm):
    assert dm.t_fund.shape == (7, 5)
    assert dm.t_price.shape == (1881, 3)
