#!/usr/bin/env python
"""
Main runner for the NYC Airbnb pipeline.

Usage examples:
  python main.py basic_cleaning
  python main.py train_val_test_split
  python main.py basic_cleaning,train_val_test_split,train_random_forest
  python main.py                  # uses config.yaml -> main.steps (supports "all")
"""

from __future__ import annotations

import os
import sys
import logging
from typing import List, Dict, Any
import importlib

import yaml

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
        # default full order
        return ["basic_cleaning", "train_val_test_split", "train_random_forest"]

    parts = [s.strip() for s in steps_str.replace(" ", ",").split(",") if s.strip()]
    if not parts:
        raise ValueError("No steps specified (empty).")
    return parts


def _import_and_run(module_path: str, func_name: str, params: Dict[str, Any]) -> None:
    """
    Import `func_name` from `module_path` and call with params.
    Import is done lazily so missing modules only error when that step runs.
    """
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    fn(**params)


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
            # sensible defaults if block is missing
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
