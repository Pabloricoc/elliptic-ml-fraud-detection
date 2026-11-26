"""Train two temporal baselines and evaluate the validation-selected model."""

import argparse
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .inference import predict_scores as predict_scores
from .ingestion import load_dataset
from .model_evaluation import (
    at_threshold,
    average_precision,
    review_queue,
    validation_threshold,
)
from .model_features import FEATURE_AUDIT, WINDOWS, Window, build_window
from .runs import (
    file_digest,
    file_state,
    implementation_digest,
    new_run,
    snapshot_digest,
    write_json,
)
from .schema import EXPECTED_FILES, DatasetError, missing_files

SEED = 42
DEFAULT_BUDGET = 50
LOGISTIC_PARAMS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": SEED}
BOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
    "random_state": SEED,
}


def _candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(**LOGISTIC_PARAMS)),
            ]
        ),
        "xgboost": Pipeline([("classifier", xgboost.XGBClassifier(**BOOST_PARAMS))]),
    }


@dataclass
class Selection:
    name: str
    model: Pipeline
    threshold: float
    candidate_ap: dict[str, float]
    operating_point: dict[str, float | int | None]


def fit_and_select(train: Window, validation: Window) -> Selection:
    """Fit solely on known training outcomes. Select using known validation outcomes."""
    if train.last >= validation.first:
        raise DatasetError("Training must precede validation in whole time steps.")
    train_y, valid_y = train.known_target(), validation.known_target()
    if train_y.nunique() != 2 or valid_y.nunique() != 2:
        raise DatasetError(
            "Training and validation each require labelled illicit and licit transactions."
        )
    train_x = train.features.loc[train_y.index]
    valid_x = validation.features.loc[valid_y.index]
    models = _candidates()
    scores = {}
    candidate_ap = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        scores[name] = predict_scores(model, valid_x)
        ap = average_precision(valid_y, scores[name])
        assert ap is not None  # Both labelled classes were required above.
        candidate_ap[name] = ap
    # Insertion order favours logistic regression on an exact AP tie.
    winner = max(candidate_ap, key=candidate_ap.__getitem__)
    threshold = validation_threshold(valid_y, scores[winner])
    return Selection(
        winner,
        models[winner],
        threshold,
        candidate_ap,
        at_threshold(valid_y, scores[winner], threshold),
    )


def run_training(
    data_dir: Path,
    output_dir: Path,
    *,
    budget_per_step: int = DEFAULT_BUDGET,
    dataset_version: str | None = None,
) -> Path:
    """Train, select, score the final window once and publish local aggregate artifacts."""
    if budget_per_step < 1:
        raise DatasetError("Review budget must be positive.")
    if output_dir.resolve().is_relative_to(data_dir.resolve()):
        raise DatasetError(
            "Choose an output directory outside the raw input directory."
        )
    missing = missing_files(data_dir)
    if missing:
        raise DatasetError(f"Missing input files in {data_dir}: {', '.join(missing)}.")
    paths = [data_dir / name for name in EXPECTED_FILES]
    before = file_state(paths)
    dataset = load_dataset(data_dir)
    fingerprints = {path.name: file_digest(path) for path in paths}
    if before != file_state(paths):
        raise DatasetError("An input file changed during ingestion or fingerprinting.")
    train = build_window(dataset, *WINDOWS["train"])
    validation = build_window(dataset, *WINDOWS["validation"])
    selection = fit_and_select(train, validation)

    test = build_window(dataset, *WINDOWS["test"])
    if validation.last >= test.first:
        raise DatasetError("The final test window must follow validation.")
    test_scores = predict_scores(selection.model, test.features)
    test_y = test.known_target()
    queue, by_step = review_queue(test, test_scores, budget_per_step, SEED)
    metrics = {
        "selected_model": selection.name,
        "threshold": selection.threshold,
        "windows": {
            name: window.support()
            for name, window in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "validation": {
            "candidate_average_precision": selection.candidate_ap,
            "operating_point": selection.operating_point,
        },
        "test": {
            "average_precision": average_precision(test_y, test_scores),
            "labelled_prevalence": float(test_y.mean()) if len(test_y) else None,
            "operating_point": at_threshold(test_y, test_scores, selection.threshold),
            "review_queue": queue,
        },
    }
    with new_run(output_dir, "model") as (staging, destination):
        artifact = staging / "model.joblib"
        joblib.dump(selection.model, artifact, compress=3)
        by_step.to_csv(staging / "test_by_step.csv")
        write_json(staging / "metrics.json", metrics)
        versions = {
            name: version(name)
            for name in (
                "elliptic-fraud",
                "joblib",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "threadpoolctl",
            )
        }
        write_json(
            staging / "manifest.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "dataset": {
                    "source": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
                    "release": "Original Elliptic transaction graph",
                    "kaggle_version": dataset_version,
                    "files_sha256": fingerprints,
                    "snapshot_sha256": snapshot_digest(fingerprints),
                },
                "implementation_sha256": implementation_digest(),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {**versions, "xgboost": xgboost.__version__},
                },
                "seed": SEED,
                "windows_inclusive": WINDOWS,
                "features_in_order": FEATURE_AUDIT,
                "availability": "At confirmation, within-window observed inputs and their ancestors only. "
                "The released subgraph is a retrospective sample.",
                "label_policy": "1=illicit (positive), 2=licit (negative). Unknown is context and queue "
                "eligible only. Earlier-window labels are assumed to have matured.",
                "candidates": {
                    "logistic_regression": {
                        "preprocessing": "StandardScaler",
                        **LOGISTIC_PARAMS,
                    },
                    "xgboost": BOOST_PARAMS,
                },
                "selection": "Unit weights. Highest labelled validation AP, then logistic on a tie. "
                "Highest threshold attaining maximum validation F1. No refit.",
                "winner": selection.name,
                "threshold": selection.threshold,
                "review_budget_per_step": budget_per_step,
                "ranking": "Descending illicit score, then ascending SHA256 of ASCII seed|tx_id.",
                "model_sha256": file_digest(artifact),
            },
        )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pre-specified temporal experiment against local CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--budget-per-step",
        type=int,
        default=DEFAULT_BUDGET,
        help="fixed review capacity for each test time step (default: 50)",
    )
    parser.add_argument(
        "--dataset-version", help="known Kaggle release version, if available"
    )
    args = parser.parse_args(argv)
    try:
        destination = run_training(
            args.data_dir,
            args.output_dir,
            budget_per_step=args.budget_per_step,
            dataset_version=args.dataset_version,
        )
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Training run written to {destination}")
    return 0
