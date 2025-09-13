import argparse
import pandas as pd
import wandb

def go(input_artifact, output_artifact, output_type, output_description, sample_size):
    run = wandb.init(job_type="make_sample_from_artifact")
    local_path = run.use_artifact(input_artifact).file()
    df = pd.read_csv(local_path)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    out = "sample.csv"
    df.to_csv(out, index=False)
    art = wandb.Artifact(name=output_artifact, type=output_type, description=output_description)
    art.add_file(out)
    run.log_artifact(art)
    run.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_artifact", required=True)      # e.g., clean_sample.csv:latest
    p.add_argument("--output_artifact", default="sample.csv")
    p.add_argument("--output_type", default="raw_data")
    p.add_argument("--output_description", default="Sample derived from clean_sample.csv")
    p.add_argument("--sample_size", type=int, default=5000)
    args = p.parse_args()
    go(**vars(args))
