#!/usr/bin/env python
"""
Main runner for the NYC Airbnb pipeline.

Usage examples:
  python main.py basic_cleaning
  python main.py train_val_test_split
  python main.py train_random_forest
  python main.py test_regression_model
  python main.py basic_cleaning,train_val_test_split,train_random_forest
  python main.py                  # uses config.yaml -> main.steps (supports "all")
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import List, Dict, Any
import importlib

import yaml

# Added for Issue 8 (and evaluate/test steps)
import joblib
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------- helpers --------

def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_wandb_project(cfg: Dict[str, Any]) -> None:
    if not os.environ.get("WANDB_PROJECT"):
        proj = cfg.get("main", {}).get("project_name")
        if proj:
            os.environ["WANDB_PROJECT"] = str(proj)
            logger.info("WANDB_PROJECT set to %s", proj)


def _parse_steps(arg_steps: str | None, cfg: Dict[str, Any]) -> List[str]:
    """
    Determine which steps to run from CLI or config.
    - CLI: comma/space-separated string like "basic_cleaning,train_val_test_split"
    - Config: main.steps can be "all" or a comma/space-separated list.
    """
    if arg_steps and arg_steps.strip():
        steps_str = arg_steps.strip()
    else:
        steps_str = str(cfg.get("main", {}).get("steps", "all")).strip()

    if steps_str.lower() in {"all", "*"}:
        # default full order for "all"
        return ["basic_cleaning", "train_val_test_split", "train_random_forest"]

    parts = [s.strip() for s in steps_str.replace(" ", ",").split(",") if s.strip()]
    if not parts:
        raise ValueError("No steps specified (empty).")
    return parts


def _import_and_run(module_path: str, func_name: str, params: Dict[str, Any]) -> None:
    """
    Import `func_name` from `module_path` and call with params.
    """
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    fn(**params)


def _find_model_paths(model_dir: str) -> tuple[str, str]:
    """
    Returns (model_path, columns_json_path), trying both flattened artifact layout and
    a nested 'model_export/' layout. Robust across W&B versions.
    """
    candidates = [
        (os.path.join(model_dir, "model.joblib"),
         os.path.join(model_dir, "columns.json")),
        (os.path.join(model_dir, "model_export", "model.joblib"),
         os.path.join(model_dir, "model_export", "columns.json")),
    ]
    for m, c in candidates:
        if os.path.exists(m) and os.path.exists(c):
            return m, c
    # Fallback: first existing model/joblib in tree
    for root, _, files in os.walk(model_dir):
        if "model.joblib" in files:
            m = os.path.join(root, "model.joblib")
            c = os.path.join(root, "columns.json")
            if os.path.exists(c):
                return m, c
    raise FileNotFoundError(f"Could not locate model.joblib/columns.json under: {model_dir}")


# -------- Issue 8: test_regression_model --------

def _test_regression_model(model_artifact: str = "random_forest_export:latest",
                           test_artifact: str = "test_data.csv:latest",
                           tolerance: float = 0.15) -> None:
    """
    Verifies that performance on the held-out test set is close to validation performance.

    - Downloads the model artifact and reads val_mae from columns.json (saved at train time).
    - Computes test_mae on the latest test split.
    - Logs both to W&B summary and asserts |test_mae - val_mae| / val_mae <= tolerance.
    """
    run = wandb.init(job_type="test_regression_model")

    # Pull artifacts
    model_dir = run.use_artifact(model_artifact).download()
    test_csv  = run.use_artifact(test_artifact).file()
    logger.info("Model dir: %s | Test CSV: %s", model_dir, test_csv)

    # Locate model + metadata robustly
    model_path, cols_path = _find_model_paths(model_dir)

    with open(cols_path) as f:
        meta = json.load(f)
    val_mae = float(meta.get("val_mae"))

    pipe = joblib.load(model_path)
    df   = pd.read_csv(test_csv)
    y, X = df["price"].values, df.drop(columns=["price"])
    test_mae = float(mean_absolute_error(y, pipe.predict(X)))

    # Log to summary
    wandb.summary["val_mae"]  = val_mae
    wandb.summary["test_mae"] = test_mae

    rel_diff = abs(test_mae - val_mae) / max(val_mae, 1e-8)
    logger.info("val_mae=%.4f  test_mae=%.4f  rel_diff=%.2f%%",
                val_mae, test_mae, 100 * rel_diff)

    assert rel_diff <= tolerance, (
        f"Test MAE {test_mae:.3f} differs from Val MAE {val_mae:.3f} "
        f"by {rel_diff:.1%} (> {tolerance:.0%})"
    )

    run.finish()
    logger.info("Test-set verification passed (within %.0f%%).", 100 * tolerance)


# -------- step dispatcher --------

def run_step(step: str, cfg: Dict[str, Any]) -> None:
    """
    Dispatch a single step by name.
    """
    # Normalize aliases
    if step == "data_split":
        step = "train_val_test_split"
    if step == "train":
        step = "train_random_forest"

    if step == "basic_cleaning":
        params = cfg.get("basic_cleaning")
        if not params:
            raise KeyError("Missing 'basic_cleaning' section in config.yaml")
        logger.info("Running step: basic_cleaning with params: %s", params)
        _import_and_run("src.basic_cleaning.run", "go", params)
        logger.info("Completed: basic_cleaning")
        return

    if step == "train_val_test_split":
        params = cfg.get("data_split")
        if not params:
            raise KeyError("Missing 'data_split' section in config.yaml")
        logger.info("Running step: train_val_test_split with params: %s", params)
        _import_and_run("src.data_split.run", "go", params)
        logger.info("Completed: train_val_test_split")
        return

    if step == "train_random_forest":
        params = cfg.get("train_random_forest")
        if not params:
            # defaults if block missing
            params = {
                "train_artifact": "trainval_data.csv:latest",
                "output_artifact": "random_forest_export",
                "val_size": 0.2,
                "random_seed": 42,
                "stratify_by": "neighbourhood_group",
            }
        logger.info("Running step: train_random_forest with params: %s", params)
        _import_and_run("src.train_random_forest.run", "go", params)
        logger.info("Completed: train_random_forest")
        return

    if step in {"test_regression_model", "verify_test"}:
        # Configurable tolerance via env if desired (e.g., TOLERANCE=0.2 python main.py test_regression_model)
        tol = float(os.environ.get("TOLERANCE", "0.15"))
        params = {
            "model_artifact": "random_forest_export:latest",
            "test_artifact": "test_data.csv:latest",
            "tolerance": tol,
        }
        logger.info("Running step: test_regression_model with params: %s", params)
        _test_regression_model(**params)
        logger.info("Completed: test_regression_model")
        return

    raise ValueError(f"Unknown step: {step}")


def main() -> None:
    cfg = _load_config("config.yaml")
    _ensure_wandb_project(cfg)

    # Accept a single CLI arg with step(s), e.g. "basic_cleaning,train_val_test_split"
    arg_steps = sys.argv[1] if len(sys.argv) > 1 else None
    steps = _parse_steps(arg_steps, cfg)

    logger.info("Steps to run: %s", steps)
    for s in steps:
        run_step(s, cfg)

    logger.info("All requested steps completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)
