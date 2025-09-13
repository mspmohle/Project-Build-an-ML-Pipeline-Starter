#!/usr/bin/env python
"""
Train a RandomForest regressor on the Airbnb dataset.

Inputs (W&B Artifacts):
- --trainval_artifact: the train/validation CSV artifact produced by data_split
                       (e.g., trainval_data.csv:latest)

Other params:
- --val_size: fraction of trainval used for validation
- --random_seed: integer seed for reproducibility
- --stratify_by: column name for stratification (use "none" to disable)
- --rf_config: path to a JSON file with RandomForestRegressor kwargs
- --max_tfidf_features: int, max features for TFIDF on the 'name' column
- --output_artifact: name of the model export artifact to log to W&B
                     (e.g., random_forest_export)
"""

import argparse
import json
import logging
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import wandb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest on Airbnb data")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        required=True,
        help="Train/validation CSV artifact name (e.g., trainval_data.csv:latest)",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        required=True,
        help="Validation split size (fraction of trainval)",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed for splits and model",
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        required=True,
        help='Column to stratify by, or "none"',
    )

    parser.add_argument(
        "--rf_config",
        type=str,
        required=True,
        help="Path to JSON file with RandomForestRegressor parameters",
    )

    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        required=True,
        help="Max features for TFIDF on the name/title column",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name for the exported model artifact (e.g., random_forest_export)",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def build_preprocessor(max_tfidf_features: int) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - one-hot encodes selected categorical columns
      - imputes numeric columns with median
      - vectorizes the text 'name' column with TF-IDF
    """
    categorical_features = ["neighbourhood_group", "room_type"]
    numeric_features = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    text_feature = "name"

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # TfidfVectorizer consumes a 1D sequence; we pass the column name to ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
            ("text", TfidfVectorizer(max_features=max_tfidf_features, stop_words="english"), text_feature),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return preprocessor


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def go(args):
    # Init W&B
    run = wandb.init(job_type="train_random_forest", save_code=True)
    run.config.update(
        {
            "trainval_artifact": args.trainval_artifact,
            "val_size": args.val_size,
            "random_seed": args.random_seed,
            "stratify_by": args.stratify_by,
            "rf_config": args.rf_config,
            "max_tfidf_features": args.max_tfidf_features,
            "output_artifact": args.output_artifact,
        }
    )

    # -------------------------------------------------------------------------
    # 1) Fetch train/val data from W&B
    # -------------------------------------------------------------------------
    logger.info("Downloading train/val artifact: %s", args.trainval_artifact)
    trainval_path = run.use_artifact(args.trainval_artifact).file()
    df = pd.read_csv(trainval_path)

    # Target & features
    target_col = "price"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Ensure text column is string and NA-free for TF-IDF
    if "name" in X.columns:
        X["name"] = X["name"].fillna("").astype(str)

    # -------------------------------------------------------------------------
    # 2) Split into train/val
    # -------------------------------------------------------------------------
    stratify = None
    if args.stratify_by.lower() != "none":
        if args.stratify_by not in df.columns:
            raise ValueError(
                f"Requested stratify_by='{args.stratify_by}', but column not in data. "
                f"Available: {list(df.columns)}"
            )
        stratify = df[args.stratify_by]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=stratify if stratify is not None else None,
    )

    # -------------------------------------------------------------------------
    # 3) Load RF params
    # -------------------------------------------------------------------------
    logger.info("Loading RF config from: %s", args.rf_config)
    with open(args.rf_config) as f:
        rf_params = json.load(f)

    # -------------------------------------------------------------------------
    # 4) Build pipeline: preprocessor + regressor (final estimator!)
    # -------------------------------------------------------------------------
    preprocessor = build_preprocessor(args.max_tfidf_features)
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**rf_params)),
        ]
    )

    # -------------------------------------------------------------------------
    # 5) Fit & evaluate
    # -------------------------------------------------------------------------
    logger.info("Fitting pipeline...")
    sk_pipe.fit(X_train, y_train)

    r2 = sk_pipe.score(X_val, y_val)
    preds = sk_pipe.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    wandb.summary["r2"] = float(r2)
    wandb.summary["rmse"] = float(rmse)
    logger.info("Validation R^2: %.4f | RMSE: %.4f", r2, rmse)

    # -------------------------------------------------------------------------
    # 6) Export model and log as W&B artifact
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = os.path.join(tmpdir, "model_export")
        os.makedirs(export_dir, exist_ok=True)
        model_path = os.path.join(export_dir, "model.pkl")
        joblib.dump(sk_pipe, model_path)

        artifact = wandb.Artifact(
            name=args.output_artifact,
            type="model_export",
            description="RandomForestRegressor pipeline (preprocessing + regressor)",
            metadata={
                "r2": float(r2),
                "rmse": float(rmse),
                "rf_params": rf_params,
                "val_size": float(args.val_size),
                "random_seed": int(args.random_seed),
                "stratify_by": args.stratify_by,
                "max_tfidf_features": int(args.max_tfidf_features),
            },
        )
        artifact.add_dir(export_dir)
        run.log_artifact(artifact)

    logger.info("Model export logged as artifact: %s", args.output_artifact)


if __name__ == "__main__":
    go(parse_args())
