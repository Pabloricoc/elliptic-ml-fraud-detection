# Working on Elliptic ML Fraud Detection

## Code and checks

- Target Python 3.11 and 3.12. Keep package code in `src/elliptic_fraud/` and tests in `tests/`. Use type annotations on public functions and `pathlib` for cross-platform paths.
- Introduce a dependency when an implemented feature uses it. Keep test and lint tools in the `dev` extra. Update `constraints-dev.txt` and verify a fresh editable install when changing the development toolchain.
- Before proposing a commit, run `python -m ruff check .`, `python -m ruff format --check .` and `python -m pytest` in the development environment. Add tests for changed behaviour through the package's normal entry points. The default suite must run offline without the dataset.
- Keep commits small. Group one change with its tests and documentation, use an imperative subject and inspect the staged diff.

## Data and ML decisions

- Preserve `unknown` as unlabelled. Never map it to licit or use it as a negative example in training or evaluation.
- Split by whole time steps before fitting. Never mix future observations into training, including through preprocessing, feature selection, neighbour aggregation or resampling. Use validation for model and threshold selection. Reserve the final time window for evaluation.
- Record the dataset version, split cutoffs, feature availability, label policy, seed and dependency versions with each experiment. Publish performance claims only after measuring them on the documented split. Follow the [evaluation protocol](docs/evaluation.md).
- Never commit or publish secrets, Kaggle credentials, raw data, processed records or dataset samples. Keep records out of notebooks, logs, test fixtures and documentation. Consult the [dataset guide](data/README.md) for provenance and licensing.
