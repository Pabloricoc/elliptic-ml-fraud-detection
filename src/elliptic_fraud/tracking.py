"""Import completed Elliptic experiments into a local MLflow SQLite store."""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from .runs import file_digest, read_json, snapshot_digest
from .schema import DatasetError

EXPERIMENT = "Elliptic ML Fraud Detection"


@dataclass
class _Report:
    params: dict
    metrics: dict
    documents: dict
    digest: str
    model_digest: str


def _read_reports(model_run: Path, score_run: Path | None) -> _Report:
    manifest_path = model_run / "manifest.json"
    manifest = read_json(manifest_path)
    metrics = read_json(model_run / "metrics.json")
    hashes = {
        "training_manifest": file_digest(manifest_path),
        "metrics": file_digest(model_run / "metrics.json"),
    }
    if file_digest(model_run / "model.joblib") != manifest.get("model_sha256"):
        raise DatasetError("Model SHA-256 differs from its training manifest.")
    if metrics.get("selected_model") != manifest.get("winner") or metrics.get(
        "threshold"
    ) != manifest.get("threshold"):
        raise DatasetError(
            "Training metrics and manifest disagree on the selected model or threshold."
        )
    documents = {"training_manifest.json": manifest, "training_metrics.json": metrics}
    try:
        params = {
            "model": manifest["winner"],
            "seed": manifest["seed"],
            "threshold": manifest["threshold"],
            "budget_per_step": manifest["review_budget_per_step"],
            "training_data_sha256": manifest["dataset"]["snapshot_sha256"],
            "features": ",".join(manifest["features_in_order"]),
            **{
                f"steps.{name}": f"{start}-{end}"
                for name, (start, end) in manifest["windows_inclusive"].items()
            },
            **{
                f"winner.{key}": value
                for key, value in manifest["candidates"][manifest["winner"]].items()
            },
        }
        observed = {
            **{
                f"validation.ap.{name}": value
                for name, value in metrics["validation"][
                    "candidate_average_precision"
                ].items()
            },
            "test.average_precision": metrics["test"]["average_precision"],
            "test.labelled_prevalence": metrics["test"]["labelled_prevalence"],
            **{f"test.{k}": v for k, v in metrics["test"]["operating_point"].items()},
            **{
                f"test.queue.{k}": v for k, v in metrics["test"]["review_queue"].items()
            },
        }
        if score_run is not None:
            batch = read_json(score_run / "summary.json")
            if (
                batch["training_manifest_sha256"] != hashes["training_manifest"]
                or batch["model_sha256"] != manifest["model_sha256"]
            ):
                raise DatasetError(
                    "Batch and training run have different model provenance."
                )
            hashes["batch"] = file_digest(score_run / "summary.json")
            hashes["importance_plot"] = file_digest(score_run / "shap_importance.png")
            documents["batch_summary.json"] = batch
            params["batch.steps"] = f"{batch['first_step']}-{batch['last_step']}"
            observed.update(
                {
                    f"batch.{k}": batch[k]
                    for k in (
                        "scored",
                        "above_threshold",
                        "queued",
                        "max_score_reconstruction_error",
                    )
                }
            )
            observed.update(
                {
                    f"shap.mean_abs.{name}": value
                    for name, value in batch["mean_abs_shap"].items()
                }
            )
    except (KeyError, TypeError) as error:
        raise DatasetError("Incomplete training or scoring report.") from error
    return _Report(
        params, observed, documents, snapshot_digest(hashes), manifest["model_sha256"]
    )


def track_experiment(
    model_run: Path, tracking_dir: Path, score_run: Path | None = None
) -> str:
    """Record parameters, frozen evaluation and optional aggregate batch explanations."""
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    root = tracking_dir.resolve()
    sources = [model_run] if score_run is None else [model_run, score_run]
    if any(root.is_relative_to(path.resolve()) for path in sources):
        raise DatasetError(
            "Choose a tracking directory outside the completed input runs."
        )
    report = _read_reports(model_run, score_run)
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    artifacts = (root / "artifacts").as_uri()
    # Serialize migrations and search/create so retried or concurrent imports of
    # the same completed reports resolve to one finished run.
    with FileLock(root / "import.lock", timeout=120):
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment is None:
            experiment_id = client.create_experiment(
                EXPERIMENT, artifact_location=artifacts
            )
        else:
            if experiment.artifact_location.rstrip("/") != artifacts.rstrip("/"):
                raise DatasetError(
                    "The experiment's artifact directory differs from the local tracking directory."
                )
            experiment_id = experiment.experiment_id
        previous = client.search_runs(
            [experiment_id], filter_string=f"tags.import_digest = '{report.digest}'"
        )
        for attempt in previous:
            if attempt.info.status == "FINISHED":
                return attempt.info.run_id
            if attempt.info.status == "RUNNING":
                # With the local import lock held, a running attempt for the
                # same reports belongs to an interrupted process.
                client.set_terminated(attempt.info.run_id, status="KILLED")
        run_id = client.create_run(
            experiment_id,
            run_name=report.params["model"],
            tags={
                "import_digest": report.digest,
                "model_sha256": report.model_digest,
                "mlflow_version": version("mlflow"),
            },
        ).info.run_id
        try:
            now = int(time.time() * 1000)
            client.log_batch(
                run_id,
                params=[Param(k, str(v)) for k, v in report.params.items()],
                metrics=[
                    Metric(k, float(v), now, 0)
                    for k, v in report.metrics.items()
                    if v is not None
                ],
            )
            for name, aggregate in report.documents.items():
                client.log_dict(run_id, aggregate, name)
            if score_run is not None:
                client.log_artifact(run_id, str(score_run / "shap_importance.png"))
            client.set_terminated(run_id, status="FINISHED")
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        return run_id


def main(argv: Sequence[str] | None = None) -> int:
    """Track a saved experiment without fitting or evaluating it again."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument(
        "--score-run", type=Path, help="optional batch generated from this model"
    )
    parser.add_argument("--tracking-dir", type=Path, default=Path("artifacts/tracking"))
    args = parser.parse_args(argv)
    try:
        run_id = track_experiment(args.model_run, args.tracking_dir, args.score_run)
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"MLflow run {run_id} in {args.tracking_dir.resolve()}")
    return 0
