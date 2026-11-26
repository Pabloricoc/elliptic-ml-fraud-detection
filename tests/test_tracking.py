import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from elliptic_fraud.schema import DatasetError
from elliptic_fraud.scoring import score_batch
from elliptic_fraud.tracking import EXPERIMENT, track_experiment
from elliptic_fraud.train import run_training


def test_mlflow_cli_logs_matching_aggregates_and_reuses_completed_import(
    model_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    from mlflow import MlflowClient

    training = run_training(model_input, tmp_path / "models")
    batch = score_batch(training, model_input, tmp_path / "scores")
    store = tmp_path / "tracking with spaces"
    command = shutil.which("elliptic-track", path=sysconfig.get_path("scripts"))
    assert command is not None
    called = subprocess.run(
        [
            command,
            "--model-run",
            str(training),
            "--score-run",
            str(batch),
            "--tracking-dir",
            str(store),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert called.returncode == 0, called.stderr
    run_id = track_experiment(training, store, batch)
    assert run_id in called.stdout
    uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.run_id == run_id and run.info.status == "FINISHED"
    summary = json.loads((batch / "summary.json").read_text())
    metrics = json.loads((training / "metrics.json").read_text())
    assert run.data.tags["model_sha256"] == summary["model_sha256"]
    assert run.data.params["steps.validation"] == "30-39"
    assert (
        run.data.metrics["test.average_precision"]
        == metrics["test"]["average_precision"]
    )
    assert (
        run.data.metrics["validation.ap.xgboost"]
        == metrics["validation"]["candidate_average_precision"]["xgboost"]
    )
    assert run.data.metrics["batch.queued"] == summary["queued"]
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "training_manifest.json",
        "training_metrics.json",
        "batch_summary.json",
        "shap_importance.png",
    }
    stored = Path(client.download_artifacts(run_id, "batch_summary.json"))
    assert json.loads(stored.read_text(encoding="utf-8")) == summary
    source = batch / "summary.json"
    summary["training_manifest_sha256"] = "0" * 64
    source.write_text(json.dumps(summary))
    with pytest.raises(DatasetError, match="provenance"):
        track_experiment(training, store, batch)
    assert len(client.search_runs([experiment.experiment_id])) == 1


def test_training_only_import_recovers_after_an_interrupted_write(
    model_input: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    from mlflow import MlflowClient

    training = run_training(model_input, tmp_path / "models")
    store = tmp_path / "tracking"

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("artifact write failed")

    with monkeypatch.context() as fault:
        fault.setattr(MlflowClient, "log_dict", unavailable)
        with pytest.raises(OSError, match="write failed"):
            track_experiment(training, store)
    uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    (failed,) = client.search_runs([experiment.experiment_id])
    assert failed.info.status == "FAILED"
    # A hard process exit can leave a run marked RUNNING after its lock is released.
    client.create_run(
        experiment.experiment_id,
        tags={"import_digest": failed.data.tags["import_digest"]},
    )
    finished = track_experiment(training, store)
    assert sorted(
        r.info.status for r in client.search_runs([experiment.experiment_id])
    ) == ["FAILED", "FINISHED", "KILLED"]
    assert client.get_run(finished).info.status == "FINISHED"
    assert {a.path for a in client.list_artifacts(finished)} == {
        "training_manifest.json",
        "training_metrics.json",
    }
