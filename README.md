# Elliptic ML Fraud Detection

This portfolio project ranks Bitcoin transactions for analyst review using the original Elliptic dataset. It tests whether the funding history observable at transaction confirmation helps find transactions that Elliptic labelled illicit, given a fixed review capacity.

## Current scope

The package validates the original CSVs, produces an aggregate exploratory report, trains two chronological baselines and scores unlabelled batches with the selected model. Analysts can inspect the review queue and its SHAP explanations locally. MLflow records the experiment settings, evaluation and aggregate explanations. A FastAPI service can score a transaction on demand with the same saved pipeline and threshold.

A local data audit covered 203,769 transactions and 234,355 directed edges across all 49 time steps. Only 46,564 transactions have a known outcome. On the final ten steps, the selected XGBoost model achieved 0.0882 average precision on labelled transactions. A budget of 50 reviews per step captured 5 of 636 known illicit cases. [Experiment results and label coverage](docs/evaluation.md#recorded-local-result)

## Development setup

Use Python 3.11 to match the reference CI job. Python 3.12 is also supported. Start in the repository root.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```sh
python3.11 -m venv .venv
. .venv/bin/activate
```

Install the package in editable mode and run the same checks as CI. The constraints file pins the data, training and development dependencies. The test suite runs offline on small generated inputs.

```sh
python -m pip install -c constraints-dev.txt -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

If PowerShell script execution is restricted, run the four commands above with `.\.venv\Scripts\python.exe` in place of `python`. CI runs on Ubuntu with Python 3.11 and Windows with Python 3.12.

## Run the data pipeline

Place the three original CSVs together in `data/raw/` or pass the directory that directly contains them with `--data-dir`. The package reads the local files in place. Filenames, schemas, licensing and the source of the format are in the [dataset guide](data/README.md).

```powershell
.\.venv\Scripts\elliptic-data.exe validate --data-dir ".\data\raw"
.\.venv\Scripts\elliptic-data.exe analyze --data-dir ".\data\raw" --output-dir ".\artifacts\eda"
```

With an activated environment on any platform, the commands are `elliptic-data validate` and `elliptic-data analyze`. Both default to `data/raw/`. The phase 0 command `elliptic-check-data [directory]` remains available for a quick filename check.

Validation checks file structure, numeric values, time steps, exact one-to-one label coverage by ID, and edge endpoints within the same time step. A blocking error exits with code `1`. Statistical warnings, including constant features, incomplete periods or duplicate edge rows, accompany a successful exit (`0`). Observations are preserved. In Python, `load_dataset` performs the same ingestion and checks.

```python
from pathlib import Path
from elliptic_fraud.ingestion import load_dataset

dataset = load_dataset(Path("data/raw"))
transactions = dataset.transactions
edges = dataset.edges
```

### Read the EDA report

`analyze` prints a new run directory under `artifacts/eda/` by default. Each completed run contains aggregate outputs and stays outside version control.

| File | Interpretation |
| --- | --- |
| `summary.json` | Table shapes as [rows, columns], all three class counts, unknown fraction, shares among known labels, constant feature names and warnings. Graph counts include isolated nodes, repeated edge rows and self-loops. Degrees count directed edge rows as supplied. |
| `time_steps.csv` | One row for each of the 49 release time steps. Label coverage divides labelled transactions by all transactions in the step. Illicit share divides illicit by labelled transactions. Empty periods have zero counts. |
| `features.csv` | One row per anonymous feature with count, mean, sample standard deviation, quartiles and range. Quantiles describe the full input supplied to the command. |
| `timeline.png` | Transaction volume, known class counts, label coverage and illicit share on a common time axis. Gaps indicate undefined proportions. |

Proportions range from 0 to 1 in the tables. Undefined statistics appear as empty CSV cells or JSON `null`. The loader keeps the numeric feature matrix in memory, so allow RAM beyond the CSV parser's working space.

## Train and evaluate

Run the fixed comparison from the repository root in PowerShell.

```powershell
.\.venv\Scripts\elliptic-train.exe --data-dir ".\data\raw" --output-dir ".\artifacts\models"
```

With an activated environment on any platform, `elliptic-train` uses these same defaults. If you know the Kaggle release version, pass it with `--dataset-version`. Each run fingerprints the three input files independently of that optional identifier.

| Window | Steps | Use |
| --- | --- | --- |
| Train | 1–29 | Fit a scaled logistic regression and CPU XGBoost on labelled transactions. |
| Validation | 30–39 | Select the model by illicit-class average precision (AP), then select its alert threshold by F1. |
| Test | 40–49 | Evaluate the frozen model and threshold. Rank all transactions in each step for 50 review slots. |

Both models use four features derived from the observed transactions that supplied a transaction's inputs. The anonymous release columns include aggregates with uncertain availability at confirmation. The feature allowlist, selection rules and assumptions about the sampled graph and label timing are in the [evaluation protocol](docs/evaluation.md).

The command prints the path to a new local run in `artifacts/models/`.

| File | Contents |
| --- | --- |
| `metrics.json` | Class counts and coverage per window, validation AP for both models, selected threshold, final AP, confusion counts and review totals. |
| `test_by_step.csv` | Per-step queue size, confirmed illicit and licit hits, unlabelled reviews and recall of known illicit cases. |
| `manifest.json` | Input and model hashes, software versions, feature definitions, split boundaries, seed, model settings and selection rules. |
| `model.joblib` | The fitted winning pipeline. The manifest records its feature order and alert threshold. Load locally produced artifacts with the recorded dependency versions. |

The supervised metrics describe the labelled subset. The fixed-budget queue includes unlabelled transactions, so its confirmed hits are a lower bound on the illicit transactions reviewed. Ranking takes place at the end of each time step. The selected threshold can be applied to individual scores as they become available.

## Score and explain a batch

Use the model directory printed by `elliptic-train`. The scoring command prints the batch directory to use for tracking. With the virtual environment active, replace the example paths with your local run paths.

```powershell
$modelRun = "artifacts/models/model-<run-id>"
elliptic-score --model-run $modelRun --data-dir data/raw --first-step 40 --last-step 49
$scoreRun = "artifacts/scores/score-<run-id>"
elliptic-track --model-run $modelRun --score-run $scoreRun
mlflow server --backend-store-uri sqlite:///artifacts/tracking/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to browse the local experiment. The batch directory holds the private transaction scores and review queue. The tracker imports aggregate results and can also record an existing training run on its own. [Batch inputs, outputs and SHAP interpretation](docs/scoring.md)

## Architecture

`ingestion` and `validation` read the source tables. `analysis` and `report` produce local diagnostics. `model_features` and `model_evaluation` provide the same predictors and ranking policy to training and scoring. `scoring` uses the saved pipeline and `explanations` computes its feature contributions. `tracking` imports completed runs into a local SQLite-backed MLflow experiment. `api` loads a frozen model at startup and reuses the same feature and inference code.

## Serve the model

Start Docker Desktop in Linux container mode and build the image with `docker build --tag elliptic-fraud:local .` from the repository root. Mount the trusted training run at startup. Once running, open `http://127.0.0.1:8000/docs` for the interactive API reference. The serving guide includes the full PowerShell commands, a synthetic request and a container smoke test. [Serving guide](docs/serving.md)

## Roadmap

| Phase | Deliverable | Status |
| --- | --- | --- |
| 0 · Bootstrap | Editable package, local file check, tests and CI configuration | Implemented |
| 1 · Data | Load and validate the original schema, profile classes and directed edges, and generate local EDA reports | Validated on the full dataset |
| 2 · Modelling | Compare logistic regression and XGBoost, evaluate a fixed review budget and save the selected pipeline | Trained and evaluated on the full release |
| 3 · Risk and explanations | Batch scoring, per-transaction SHAP explanations for the review queue and local MLflow tracking | Implemented |
| 4 · Serving | Serve the frozen model with FastAPI, test score parity and run the Docker smoke test in CI | Implemented |
