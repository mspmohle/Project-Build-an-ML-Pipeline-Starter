import os
import pandas as pd

DATA_PATH = os.environ.get("DATA_PATH", "clean_data.csv")
MIN_PRICE = float(os.environ.get("MIN_PRICE", "10"))
MAX_PRICE = float(os.environ.get("MAX_PRICE", "350"))

def test_required_columns_present_and_valid():
    df = pd.read_csv(DATA_PATH)
    required = ["price", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    assert df["latitude"].between(-90, 90).all(), "Invalid latitude values present"
    assert df["longitude"].between(-180, 180).all(), "Invalid longitude values present"

def test_price_within_expected_range():
    df = pd.read_csv(DATA_PATH)
    ok = df["price"].between(MIN_PRICE, MAX_PRICE)
    assert ok.all(), f"Found {(~ok).sum()} rows with price outside [{MIN_PRICE}, {MAX_PRICE}]"
