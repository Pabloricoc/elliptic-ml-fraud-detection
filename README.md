# Elliptic ML Fraud Detection

The goal of this portfolio project is to rank Bitcoin transactions for analyst review using the original Elliptic dataset. The intended prediction is whether a transaction carries Elliptic's illicit label. The evaluation will ask how many labelled illicit transactions an analyst could find with a limited review budget.

## Current scope

Phase 1 adds ingestion, validation and exploratory analysis for the original Elliptic CSVs. The package preserves every transaction and its source class, including `unknown`, and keeps directed edges in a separate table. The automated suite uses small generated CSVs. A local run on the full Elliptic dataset passed validation and produced an aggregate report covering all 49 time steps.

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

Install the package in editable mode and run the same checks as CI. The constraints file pins the data and development dependencies. The test suite runs offline on small generated inputs.

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

## Planned architecture

The offline pipeline will join transaction features and labels by transaction ID, retain the time step, and build chronological train, validation and test partitions. Verified features and a fitted model will produce a risk score used to order the review queue. Batch scoring and the eventual API will use the same preprocessing and model artifact. The [evaluation protocol](docs/evaluation.md) defines which transactions and features can support a performance claim.

## Roadmap

| Phase | Deliverable | Status |
| --- | --- | --- |
| 0 · Bootstrap | Editable package, local file check, tests and CI configuration | Implemented |
| 1 · Data | Load and validate the original schema, profile classes and directed edges, and generate local EDA reports | Validated on the full dataset |
| 2 · Modelling | Compare a scikit-learn baseline with XGBoost on chronological partitions | Planned |
| 3 · Risk and explanations | Set a review budget, evaluate ranking, explain scores with SHAP and record runs with MLflow | Planned |
| 4 · Serving | Package the selected model behind FastAPI, test inference and build a Docker image | Planned |
