Build an ML Pipeline for NYC Short-Term Rental Prices
End-to-end, reproducible ML pipeline that downloads a data sample, cleans & validates it, splits train/val/test, trains a Random Forest regressor with MLflow + Hydra, tracks everything in Weights & Biases (W&B), and exports a production model artifact.
	•	Repo: https://github.com/mspmohle/Project-Build-an-ML-Pipeline-Starter
	•	W&B Project: https://wandb.ai/mmohle-wgu/nyc_airbnb

Quick Start (Reproduce from Release)
Requires Conda + Git. MLflow will create step environments as needed.
# Run full pipeline from the tagged release (sample1.csv)
mlflow run https://github.com/mspmohle/Project-Build-an-ML-Pipeline-Starter.git -v 1.0.0

# Re-run the same release on the new dataset sample2.csv
mlflow run https://github.com/mspmohle/Project-Build-an-ML-Pipeline-Starter.git \
  -v 1.0.0 \
  -P "hydra_options=etl.sample='sample2.csv'"
If artifacts don’t appear in W&B from child Conda envs, set:
export WANDB_API_KEY=<your key> WANDB_MODE=online
export WANDB_ENTITY=mmohle-wgu WANDB_PROJECT=nyc_airbnb

Project Overview
	•	Goal: Predict nightly price for NYC listings; retrain reproducibly on new incoming data.
	•	Stack: Python, MLflow Projects, Hydra, scikit-learn, Pandas, W&B artifacts & lineage.
	•	Why: Consistent, auditable retraining with automated data checks and model tracking.

Repository Structure
```text
├─ src/
│  ├─ basic_cleaning/run.py
│  ├─ data_check/test_data.py
│  ├─ train_random_forest/run.py
│  └─ utils/… (optional helpers)
├─ main.py
├─ configs/ (Hydra)
│  ├─ config.yaml
│  └─ component-specific configs (optional)
├─ README.md (links to GitHub + public W&B project)
└─ LICENSE, .github/workflows/ci.yml, requirements.txt

Configuration (Hydra)
Key parameters (frozen for release 1.0.0):
main:
  components_repository: "https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter.git#components"
  project_name: nyc_airbnb
  experiment_name: development
  steps: all

etl:
  sample: "sample1.csv"
  min_price: 10
  max_price: 350

data_check:
  kl_threshold: 0.2

modeling:
  test_size: 0.2
  val_size: 0.2
  random_seed: 42
  stratify_by: "neighbourhood_group"
  max_tfidf_features: 5
  output_artifact: random_forest_export
  random_forest:
    n_estimators: 200
    max_depth: 50
    min_samples_split: 4
    min_samples_leaf: 3
    n_jobs: -1
    criterion: squared_error
    max_features: 0.5
    oob_score: true
Pipeline Steps
	1	download (remote component): fetches etl.sample → logs sample.csv (raw_data).
	2	basic_cleaning: removes price outliers, parses dates, filters to NYC geo bounds → logs clean_sample.csv.
	3	data_check: schema/geo/row count/price-range tests; KL divergence vs reference.
	4	data_split (remote component): creates trainval_data.csv and test_data.csv.
	5	train_random_forest: text + numeric feature pipeline, fits RF, logs model export (MLflow format).
	6	test_regression_model (manual step): evaluates prod model on test_data.csv.
Run everything locally:
mlflow run .
Run selected steps:
mlflow run . -P steps=basic_cleaning
mlflow run . -P steps=basic_cleaning,data_check
Override at runtime (Hydra):
mlflow run . \
  -P steps=train_random_forest \
  -P "hydra_options=modeling.random_forest.max_depth=10 modeling.random_forest.n_estimators=100"
HPO (Hydra multi-run)
mlflow run . \
  -P steps=train_random_forest \
  -P "hydra_options=modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"

Winner (frozen in config.yaml): n_estimators=200, max_depth=50 (other params as above).
Best Model (Production)
	•	Artifact: random_forest_export:v5 (alias: prod) https://wandb.ai/mmohle-wgu/nyc_airbnb/artifacts/model_export/random_forest_export/v5
	•	Metrics: R² = 0.5507971202455809, RMSE = 48.14590328345455
Note: W&B may deduplicate artifacts when content is identical; subsequent release runs can validly reference the same artifact version.

Artifacts & Lineage (W&B)
	•	Project: https://wandb.ai/mmohle-wgu/nyc_airbnb
	•	Key artifacts:
	◦	sample.csv (raw_data)
	◦	clean_sample.csv
	◦	trainval_data.csv, test_data.csv
	◦	random_forest_export (model_export) — production alias prod → v5
	•	Lineage: Open the model export artifact (v5) → Lineage tab to see the full DAG (download → clean → checks → split → train).

Releases
	•	1.0.0
	◦	Freezes best Random Forest hyperparameters (see config.yaml).
	◦	Repro:
	◦	mlflow run https://github.com/mspmohle/Project-Build-an-ML-Pipeline-Starter.git -v 1.0.0
	◦	mlflow run https://github.com/mspmohle/Project-Build-an-ML-Pipeline-Starter.git \
	◦	  -v 1.0.0 \
	◦	  -P "hydra_options=etl.sample='sample2.csv'"
	◦	If the geo-boundary test fails for new samples, ensure src/basic_cleaning/run.py filters to NYC:
idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
Data Checks (what’s enforced)
	•	Expected columns and order
	•	Neighborhood names set matches known NYC boroughs
	•	NYC geo bounds (longitude/latitude)
	•	KL divergence threshold vs reference distribution
	•	Row count within expected range
	•	Price within configured min/max

Troubleshooting
	•	Artifacts not visible in W&B from release runs? Child Conda envs may be offline. Export your key before running:    export WANDB_API_KEY=<key> WANDB_MODE=online
	•	export WANDB_ENTITY=mmohle-wgu WANDB_PROJECT=nyc_airbnb
	•	  
	•	Stale MLflow envs causing odd behavior?    conda info --envs | grep mlflow | awk '{print $1}' | xargs -I{} conda remove -n {} --all -y
	•	  
Weights and Balaces Link: https://wandb.ai/mmohle-wgu/Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning?nw=nwusermmohle
MIT License

Copyright (c) 2025 Michael Mohle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.







