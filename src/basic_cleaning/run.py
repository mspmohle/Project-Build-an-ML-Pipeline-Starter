#!/usr/bin/env python3
"""
basic_cleaning: download raw dataset from W&B, clean it, and log a cleaned artifact.

Adds NYC geofence (lon ∈ [-74.25, -73.50], lat ∈ [40.5, 41.2]).
Set environment variable SKIP_GEOFENCE=1 to skip the geofence (for Issue 10 demo).
"""

import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger("basic_cleaning")


def go(input_artifact: str,
       output_artifact: str,
       output_type: str,
       output_description: str,
       min_price: float,
       max_price: float) -> None:

    run = wandb.init(job_type="basic_cleaning")
    run.config.update({
        "input_artifact": input_artifact,
        "output_artifact": output_artifact,
        "output_type": output_type,
        "output_description": output_description,
        "min_price": min_price,
        "max_price": max_price,
    })

    # 1) Pull input
    local_path = run.use_artifact(input_artifact).file()
    df = pd.read_csv(local_path)
    logger.info("Loaded %s with %d rows", input_artifact, len(df))

    # 2) Drop price outliers
    df = df[df["price"].between(min_price, max_price)].copy()

    # 3) Parse dates if present
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # 4) NYC geofence (can be skipped for Issue 10 demo)
    skip_geo = os.getenv("SKIP_GEOFENCE") == "1"
    if not skip_geo and {"longitude", "latitude"}.issubset(df.columns):
        in_lon = df["longitude"].between(-74.25, -73.50)
        in_lat = df["latitude"].between(40.5, 41.2)
        before = len(df)
        df = df[in_lon & in_lat].copy()
        logger.info("Geofence kept %d/%d rows", len(df), before)
    else:
        logger.warning("Geofence skipped (SKIP_GEOFENCE=%s)", os.getenv("SKIP_GEOFENCE"))

    # 5) Save + log
    out_csv = "clean_data.csv"
    df.to_csv(out_csv, index=False)

    art = wandb.Artifact(
        name=output_artifact,
        type=output_type,
        description=output_description
    )
    art.add_file(out_csv)
    run.log_artifact(art)
    run.finish()
    logger.info("Logged %s as %s (%s)", out_csv, output_artifact, output_type)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A very basic data cleaning")
    p.add_argument("--input_artifact", required=True, type=str, help="Input artifact (e.g., sample.csv:latest)")
    p.add_argument("--output_artifact", required=True, type=str, help="Output artifact name, e.g., clean_data.csv")
    p.add_argument("--output_type", required=True, type=str, help="Artifact type, e.g., clean_data")
    p.add_argument("--output_description", required=True, type=str, help="Short description of output")
    p.add_argument("--min_price", required=True, type=float, help="Min price to keep")
    p.add_argument("--max_price", required=True, type=float, help="Max price to keep")
    args = p.parse_args()
    go(**vars(args))
