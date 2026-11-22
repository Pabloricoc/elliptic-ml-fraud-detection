# Elliptic ML Fraud Detection

The goal of this portfolio project is to rank Bitcoin transactions for analyst review using the original Elliptic dataset. The intended prediction is whether a transaction carries Elliptic's illicit label. The evaluation will ask how many labelled illicit transactions an analyst could find with a limited review budget.

## Current scope

Phase 0 provides an installable Python package, a local dataset file check, pytest tests and a GitHub Actions workflow. Installation and tests run independently of the dataset. Training, model performance and inference are future milestones.

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

Install the package in editable mode and run the same checks as CI. The constraints file pins the development tools and their dependencies.

```sh
python -m pip install -c constraints-dev.txt -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

If PowerShell script execution is restricted, run the four commands above with `.\.venv\Scripts\python.exe` in place of `python`. CI runs on Ubuntu with Python 3.11 and Windows with Python 3.12.

## Optional local file check

With the environment active, run `elliptic-check-data` from the repository root to inspect `data/raw/`, or pass the directory containing the three original CSVs. Quote paths containing spaces.

```sh
elliptic-check-data "path/to/elliptic_bitcoin_dataset"
```

Exit code `0` means all expected names refer to regular files. Exit code `1` identifies a directory or file layout problem. The check inspects filesystem metadata only. File roles and access terms are in the [dataset guide](data/README.md).

## Planned architecture

The offline pipeline will join transaction features and labels by transaction ID, retain the time step, and build chronological train, validation and test partitions. Verified features and a fitted model will produce a risk score used to order the review queue. Batch scoring and the eventual API will use the same preprocessing and model artifact. The [evaluation protocol](docs/evaluation.md) defines which transactions and features can support a performance claim.

## Roadmap

| Phase | Deliverable | Status |
| --- | --- | --- |
| 0 · Bootstrap | Editable package, local file check, tests and CI configuration | Implemented |
| 1 · Data | Validate schemas, keys, labels and time steps with pandas and NumPy | Planned |
| 2 · Modelling | Compare a scikit-learn baseline with XGBoost on chronological partitions | Planned |
| 3 · Risk and explanations | Set a review budget, evaluate ranking, explain scores with SHAP and record runs with MLflow | Planned |
| 4 · Serving | Package the selected model behind FastAPI, test inference and build a Docker image | Planned |
