import os, pandas as pd
DATA = os.environ.get("DATA_PATH", "clean_data.csv")
def test_all_rows_in_nyc_bounds():
    df = pd.read_csv(DATA)
    assert df["longitude"].between(-74.25, -73.50).all(), "Found longitude out of NYC bounds"
    assert df["latitude"].between(40.5, 41.2).all(), "Found latitude out of NYC bounds"
