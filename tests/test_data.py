# src/data_check/test_data.py
"""
Data tests for the NYC Airbnb pipeline.

Implements:
- test_row_count: dataset isn't trivially small
- test_price_range: all prices within configured bounds
"""

import os
import pandas as pd


# Defaults can be overridden via env vars when running pytest
MIN_PRICE = int(os.environ.get("MIN_PRICE", "10"))
MAX_PRICE = int(os.environ.get("MAX_PRICE", "350"))
DATA_PATH = os.environ.get("DATA_PATH", "clean_data.csv")


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def test_row_count():
    """
    Ensure we have a reasonable number of rows after cleaning.
    Adjust the lower bound if your sample is smaller.
    """
    df = _read_csv(DATA_PATH)
    assert len(df) > 100, f"Row count too small: {len(df)} (DATA_PATH={DATA_PATH})"


def test_price_range():
    """
    Prices must be within [MIN_PRICE, MAX_PRICE] (inclusive) after cleaning.
    """
    df = _read_csv(DATA_PATH)
    assert df["price"].ge(MIN_PRICE).all(), f"Found price < {MIN_PRICE}"
    assert df["price"].le(MAX_PRICE).all(), f"Found price > {MAX_PRICE}"
