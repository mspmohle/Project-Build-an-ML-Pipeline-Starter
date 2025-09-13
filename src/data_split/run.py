import argparse, logging
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def go(input_artifact: str,
       trainval_artifact: str,
       test_artifact: str,
       output_type: str,
       output_description: str,
       test_size: float,
       val_size: float,
       random_seed: int,
       stratify_by: str = "none") -> None:
    """
    Split cleaned data into train+val and test; log both to W&B.
    """
    run = wandb.init(job_type="data_split")
    local_path = run.use_artifact(input_artifact).file()
    df = pd.read_csv(local_path)

    stratify = None
    if stratify_by and stratify_by.lower() != "none" and stratify_by in df.columns:
        stratify = df[stratify_by]

    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=stratify
    )

    df_trainval.to_csv("trainval_data.csv", index=False)
    df_test.to_csv("test_data.csv", index=False)

    for fname, name in [("trainval_data.csv", trainval_artifact),
                        ("test_data.csv", test_artifact)]:
        art = wandb.Artifact(name=name, type=output_type, description=output_description)
        art.add_file(fname)
        run.log_artifact(art)
        logger.info("Logged %s as %s", fname, name)

    run.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Data splitting step")
    p.add_argument("--input_artifact", required=True, type=str)   # e.g. clean_data.csv:latest
    p.add_argument("--trainval_artifact", default="trainval_data.csv", type=str)
    p.add_argument("--test_artifact", default="test_data.csv", type=str)
    p.add_argument("--output_type", default="processed_data", type=str)
    p.add_argument("--output_description", default="Train+Val and Test splits", type=str)
    p.add_argument("--test_size", default=0.2, type=float)
    p.add_argument("--val_size", default=0.2, type=float)  # train/val split done later in training
    p.add_argument("--random_seed", default=42, type=int)
    p.add_argument("--stratify_by", default="neighbourhood_group", type=str)
    args = p.parse_args()
    go(**vars(args))
