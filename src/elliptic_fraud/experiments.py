"""Run fixed T0 baselines on cached development windows."""

import argparse
import os
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import joblib
from filelock import FileLock

from .evaluate import evaluate_window, evaluation_environment
from .folds import DEV_STEPS, development_folds, slice_development, validate_development
from .inference import predict_scores
from .model_features import FEATURE_AUDIT, Window
from .runs import (
    file_digest,
    implementation_digest,
    read_json,
    snapshot_digest,
    write_json,
)
from .schema import DatasetError
from .train import BOOST_PARAMS, LOGISTIC_PARAMS, SEED, _candidates

FEATURE_VERSION = 1
DEVELOPMENT_THRESHOLD = 0.5
TRACKING_EXPERIMENT = "Elliptic S1 development"


def feature_code_sha256() -> str:
    """Fingerprint the code that parses, validates and builds T0 inputs."""
    return snapshot_digest(
        {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "ingestion.py",
                "schema.py",
                "validation.py",
                "model_features.py",
            )
        }
    )


def development_cache_path(snapshot_sha: str) -> Path:
    """Locate the cached T0 development-only snapshot."""
    if len(snapshot_sha) != 64 or any(
        char not in "0123456789abcdef" for char in snapshot_sha
    ):
        raise DatasetError("Expected a SHA-256 snapshot identifier.")
    return (
        Path("data/processed")
        / snapshot_sha
        / f"t0-v{FEATURE_VERSION}-{feature_code_sha256()}"
    )


def cache_development(window: Window, snapshot: dict) -> Path:
    """Cache only steps 1–39 while the approved full snapshot is already in memory."""
    validate_development(window)
    if list(window.features.columns) != list(FEATURE_AUDIT):
        raise DatasetError("Development cache requires the frozen T0 feature order.")
    root = development_cache_path(snapshot["snapshot_sha256"]).resolve()
    if not root.is_relative_to(Path("data/processed").resolve()):
        raise DatasetError(
            "Development cache must be inside the local processed-data directory."
        )
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".cache.lock"):
        manifest = root / "manifest.json"
        if not manifest.exists():
            payload = root / "development.joblib"
            temporary = root / ".development.partial"
            joblib.dump(window, temporary, compress=3)
            temporary.replace(payload)
            write_json(
                root / ".manifest.partial",
                {
                    "dataset": snapshot,
                    "feature_set": "T0",
                    "feature_version": FEATURE_VERSION,
                    "feature_code_sha256": feature_code_sha256(),
                    "steps": DEV_STEPS,
                    "cache_sha256": file_digest(payload),
                },
            )
            (root / ".manifest.partial").replace(manifest)
    return root


def _load_development(root: Path) -> tuple[Window, dict]:
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("steps") != list(DEV_STEPS)
        or manifest.get("feature_set") != "T0"
        or manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("feature_code_sha256") != feature_code_sha256()
    ):
        raise DatasetError(
            "The cache is not compatible with the development-only T0 contract."
        )
    payload = root / "development.joblib"
    if file_digest(payload) != manifest.get("cache_sha256"):
        raise DatasetError("The development cache does not match its manifest.")
    window = joblib.load(payload)
    if not isinstance(window, Window) or list(window.features.columns) != list(
        FEATURE_AUDIT
    ):
        raise DatasetError("The development cache does not contain a T0 window.")
    validate_development(window)
    return window, manifest


def record_fold_mlflow(output_dir: Path, report: dict) -> str:
    """Track one aggregate fold report in its scenario's local experiment."""
    # Use a separate SQLite database and aggregate artifacts for development.
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    from mlflow import MlflowClient

    tracking = (output_dir / "tracking").resolve()
    tracking.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{(tracking / 'mlflow.db').as_posix()}"
    config_hash = report["config_sha256"]
    name = (
        TRACKING_EXPERIMENT
        if report["config"]["scenario"] == "S1"
        else "Elliptic S2 development"
    )
    with FileLock(tracking / ".tracking.lock"):
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment = client.get_experiment_by_name(name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                name, artifact_location=(tracking / "artifacts").as_uri()
            )
        )
        for previous in client.search_runs(
            [experiment_id], filter_string=f"tags.config_sha256 = '{config_hash}'"
        ):
            if previous.info.status == "FINISHED":
                return previous.info.run_id
            if previous.info.status == "RUNNING":
                client.set_terminated(previous.info.run_id, status="KILLED")
        config = report["config"]
        run_id = client.create_run(
            experiment_id,
            run_name=f"{config['family']}-{config['fold']['name']}",
            tags={"config_sha256": config_hash, "git_commit": report["git_commit"]},
        ).info.run_id
        try:
            for name, value in config.items():
                client.log_param(run_id, name, value)
            metrics = report["validation"]
            metrics_to_log = {
                name: metrics[name]
                for name in ("mean_step_ap", "pooled_ap", "roc_auc", "brier_score")
            }
            metrics_to_log["p_at_50_labelled"] = metrics["q_lab"]["50"][
                "mean_step_precision"
            ]
            metrics_to_log["queue_hits_over_expected"] = metrics["q_all"]["50"][
                "hits_over_expected"
            ]
            for name, value in metrics_to_log.items():
                if value is not None:
                    client.log_metric(run_id, name, value)
            client.log_dict(run_id, report, "evaluation.json")
            client.set_terminated(run_id)
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        return run_id


def run_baseline_folds(
    development_cache: Path, output_dir: Path, *, git_commit: str, embargo: int = 0
) -> Path:
    """Fit the fixed LR and XGBoost configurations once per development fold."""
    protected = [
        development_cache,
        Path("data/raw"),
        *(Path("artifacts") / name for name in ("models", "scores", "eda", "tracking")),
    ]
    output_dir = output_dir.resolve()
    if any(output_dir.is_relative_to(path.resolve()) for path in protected):
        raise DatasetError(
            "Choose an experiments directory outside protected input and artifact directories."
        )
    folds = development_folds(embargo)
    development, manifest = _load_development(development_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_sha = implementation_digest()
    environment = evaluation_environment()
    report_rows = []
    parameters = {
        "logistic_regression": {"preprocessing": "StandardScaler", **LOGISTIC_PARAMS},
        "xgboost": BOOST_PARAMS,
    }
    for fold in folds:
        train = slice_development(development, fold.train)
        validation = slice_development(development, fold.validation)
        target = train.known_target()
        if target.nunique() != 2:
            raise DatasetError("Each training fold needs both known classes.")
        for family, model in _candidates().items():
            config = {
                "schema_version": 1,
                "scenario": "S1",
                "feature_set": "T0",
                "feature_version": FEATURE_VERSION,
                "feature_code_sha256": manifest["feature_code_sha256"],
                "implementation_sha256": implementation_sha,
                "environment": environment,
                "family": family,
                "params": parameters[family],
                "seed": SEED,
                "fold": {
                    **asdict(fold),
                    "inner_early_stopping": fold.inner_early_stopping,
                },
                "embargo": embargo,
                "snapshot_sha256": manifest["dataset"]["snapshot_sha256"],
                "diagnostic_threshold": DEVELOPMENT_THRESHOLD,
            }
            digest = snapshot_digest(config)
            folder = output_dir / digest
            path = folder / "evaluation.json"
            with FileLock(output_dir / f".{digest}.lock"):
                if not path.exists():
                    model.fit(train.features.loc[target.index], target)
                    scores = predict_scores(model, validation.features).astype(
                        "float64"
                    )
                    validation_metrics = evaluate_window(
                        validation, scores, seed=SEED, threshold=DEVELOPMENT_THRESHOLD
                    )
                    report = {
                        "config_sha256": digest,
                        "config": config,
                        "git_commit": git_commit,
                        "implementation_sha256": implementation_sha,
                        "environment": environment,
                        "label_policy": "1 illicit, 2 licit, unknown unlabelled",
                        "feature_availability": FEATURE_AUDIT,
                        "train": train.support(),
                        "validation": validation_metrics,
                    }
                    folder.mkdir(exist_ok=True)
                    temporary = folder / ".evaluation.partial"
                    write_json(temporary, report)
                    temporary.replace(path)
                report = read_json(path)
                if report.get("config_sha256") != digest:
                    raise DatasetError(
                        "A saved fold report does not match its configuration."
                    )
            run_id = record_fold_mlflow(output_dir, report)
            report_rows.append(
                {
                    "model": family,
                    "fold": fold.name,
                    "config_sha256": digest,
                    "mean_step_ap": report["validation"]["mean_step_ap"],
                    "mlflow_run_id": run_id,
                }
            )
    models = {}
    for family in parameters:
        defined = [
            row["mean_step_ap"]
            for row in report_rows
            if row["model"] == family and row["mean_step_ap"] is not None
        ]
        models[family] = {
            "mean_step_ap_over_folds": statistics.mean(defined) if defined else None,
            "fold_ap_std": statistics.stdev(defined) if len(defined) > 1 else None,
            "defined_folds": len(defined),
        }
    summary = {"embargo": embargo, "models": models, "folds": report_rows}
    identity = snapshot_digest(
        {row["config_sha256"]: row["mlflow_run_id"] for row in report_rows}
    )
    destination = output_dir / f"walk-forward-{identity}.json"
    with FileLock(output_dir / f".{identity}.lock"):
        if not destination.exists():
            temporary = destination.with_suffix(".partial")
            write_json(temporary, summary)
            temporary.replace(destination)
    return destination


def clean_git_revision() -> str:
    """Read the current revision and reject uncommitted changes."""
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DatasetError(
            "Run the baseline command in a committed Git checkout."
        ) from error
    if status.strip():
        raise DatasetError(
            "Commit or stash working tree changes before running baselines."
        )
    return revision.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run development-only walk-forward baselines from a clean checkout."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--snapshot-sha", required=True, help="SHA-256 of the cached dataset snapshot"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments")
    )
    parser.add_argument("--embargo", type=int, choices=(0, 2), default=0)
    args = parser.parse_args(argv)
    try:
        revision = clean_git_revision()
        output = run_baseline_folds(
            development_cache_path(args.snapshot_sha),
            args.output_dir,
            git_commit=revision,
            embargo=args.embargo,
        )
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Walk-forward evaluation written to {output}")
    return 0
