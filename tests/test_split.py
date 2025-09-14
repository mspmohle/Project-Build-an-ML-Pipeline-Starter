import os
import pandas as pd
import numpy as np

TRAINVAL_PATH = os.environ.get("DATA_TRAINVAL_PATH", "trainval_data.csv")
TEST_PATH     = os.environ.get("DATA_TEST_PATH", "test_data.csv")
TEST_SIZE     = float(os.environ.get("TEST_SIZE", "0.2"))
TOL           = float(os.environ.get("TEST_PROP_TOL", "0.05"))   # ±5% tolerance
KL_THRESH     = float(os.environ.get("KL_THRESH", "0.2"))
STRAT_COL     = os.environ.get("STRATIFY_BY", "neighbourhood_group")

def _load(p):
    assert os.path.exists(p), f"Missing file: {p}"
    return pd.read_csv(p)

def _key_series(df: pd.DataFrame) -> pd.Series:
    for k in ("id", "listing_id"):
        if k in df.columns:
            return df[k].astype(str)
    return pd.util.hash_pandas_object(df.astype(str), index=False).astype(str)

def _kl_div(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def test_no_overlap():
    tr = _load(TRAINVAL_PATH); te = _load(TEST_PATH)
    k_tr = set(_key_series(tr)); k_te = set(_key_series(te))
    inter = k_tr.intersection(k_te)
    assert len(inter) == 0, f"Found {len(inter)} overlapping rows"

def test_proportions():
    tr = _load(TRAINVAL_PATH); te = _load(TEST_PATH)
    total = len(tr) + len(te)
    prop_test = len(te) / total if total else 0.0
    assert abs(prop_test - TEST_SIZE) <= TOL, \
        f"Test proportion {prop_test:.3f} not within ±{TOL} of {TEST_SIZE}"

def test_stratify_distribution():
    tr = _load(TRAINVAL_PATH); te = _load(TEST_PATH)
    if STRAT_COL not in tr.columns or STRAT_COL not in te.columns:
        return  # skip if column missing
    cats = sorted(set(tr[STRAT_COL].dropna().unique()) | set(te[STRAT_COL].dropna().unique()))
    p = tr[STRAT_COL].value_counts(normalize=True).reindex(cats, fill_value=0.0).values
    q = te[STRAT_COL].value_counts(normalize=True).reindex(cats, fill_value=0.0).values
    kl = _kl_div(p, q)
    assert kl <= KL_THRESH, f"KL divergence {kl:.3f} exceeds threshold {KL_THRESH} for {STRAT_COL}"
