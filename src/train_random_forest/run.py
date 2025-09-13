import argparse, json, logging, os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def go(train_artifact: str,
       output_artifact: str,
       val_size: float,
       random_seed: int,
       stratify_by: str = "none",
       # RandomForest hyperparams:
       n_estimators: int = 200,
       max_depth: int = 50,
       min_samples_split: int = 4,
       min_samples_leaf: int = 3,
       n_jobs: int = -1,
       criterion: str = "squared_error",
       max_features: float = 0.5) -> None:
    """
    Train a RandomForest regressor on train/val split derived from trainval_data.
    Logs MAE to W&B and uploads a 'model_export' artifact (random_forest_export by default).
    """
    run = wandb.init(job_type="train_random_forest")
    cfg = dict(train_artifact=train_artifact, output_artifact=output_artifact,
               val_size=val_size, random_seed=random_seed, stratify_by=stratify_by,
               n_estimators=n_estimators, max_depth=max_depth,
               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
               n_jobs=n_jobs, criterion=criterion, max_features=max_features)
    run.config.update(cfg)

    # 1) Pull trainval_data from W&B
    logger.info("Downloading input artifact: %s", train_artifact)
    local_path = run.use_artifact(train_artifact).file()
    df = pd.read_csv(local_path)

    # 2) Split into train/val
    y = df["price"].values
    X = df.drop(columns=["price"])
    stratify = None
    if stratify_by and stratify_by.lower() != "none" and stratify_by in df.columns:
        stratify = df[stratify_by]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_seed, stratify=stratify
    )

    # 3) Build preprocessing
    #   - categorical: most_frequent imputation + one-hot
    #   - numeric: median imputation (RF is scale-invariant; scaler not needed)
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("cat", categorical, cat_cols),
            ("num", numeric, num_cols),
        ]
    )

    # 4) Model pipeline
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        criterion=criterion,
        max_features=max_features,
        random_state=random_seed,
    )

    sk_pipe = Pipeline(steps=[("preprocess", preproc), ("model", rf)])

    # 5) Fit & evaluate
    logger.info("Fitting model...")
    sk_pipe.fit(X_train, y_train)
    preds = sk_pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))
    logger.info("Validation MAE: %.4f", mae)
    wandb.summary["val_mae"] = mae

    # 6) Export model files
    os.makedirs("model_export", exist_ok=True)
    joblib.dump(sk_pipe, "model_export/model.joblib")
    meta = {
        "features": {
            "categorical": cat_cols,
            "numerical": num_cols
        },
        "target": "price",
        "val_mae": mae
    }
    with open("model_export/columns.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 7) Log artifact
    art = wandb.Artifact(output_artifact, type="model_export",
                         description="RandomForest model and preprocessing pipeline")
    art.add_dir("model_export")
    run.log_artifact(art)
    run.finish()
    logger.info("Logged model artifact '%s' (type=model_export)", output_artifact)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Random Forest")
    p.add_argument("--train_artifact", type=str, required=True)       # e.g. trainval_data.csv:latest
    p.add_argument("--output_artifact", type=str, default="random_forest_export")
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--stratify_by", type=str, default="neighbourhood_group")
    # RF hyperparams
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=50)
    p.add_argument("--min_samples_split", type=int, default=4)
    p.add_argument("--min_samples_leaf", type=int, default=3)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--criterion", type=str, default="squared_error")
    p.add_argument("--max_features", type=float, default=0.5)
    args = p.parse_args()
    go(**vars(args))
