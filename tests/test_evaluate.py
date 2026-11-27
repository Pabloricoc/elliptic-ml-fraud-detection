import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from elliptic_fraud.evaluate import evaluate_legacy
from elliptic_fraud.runs import file_digest, read_json
from elliptic_fraud.schema import DatasetError
from elliptic_fraud.scoring import score_batch
from elliptic_fraud.train import run_training


def test_installed_evaluator_replays_frozen_model_and_keeps_reports_aggregate(
    model_input: Path, tmp_path: Path
) -> None:
    training = run_training(model_input, tmp_path / "trained", budget_per_step=2)
    scores = score_batch(training, model_input, tmp_path / "scores")
    inputs = {
        path: file_digest(path)
        for folder in (training, scores)
        for path in folder.iterdir()
    }
    existing = read_json(training / "metrics.json")["test"]
    command = shutil.which("elliptic-evaluate", path=sysconfig.get_path("scripts"))
    assert command is not None
    output = tmp_path / "evaluations"
    call = subprocess.run(
        [
            command,
            "--data-dir",
            str(model_input),
            "--model-run",
            str(training),
            "--score-run",
            str(scores),
            "--output-dir",
            str(output),
            "--prepare-development",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert call.returncode == 0, call.stderr
    (run,) = output.iterdir()
    report = read_json(run / "evaluation.json")
    assert report["score_run_verified"]
    assert report["test"]["pooled_ap"] == existing["average_precision"]
    assert report["test"]["operating_point"] == existing["operating_point"]
    actual = report["test"]["q_all"]["2"]
    assert {key: actual[key] for key in existing["review_queue"]} == existing[
        "review_queue"
    ]
    assert {path: file_digest(path) for path in inputs} == inputs
    assert "000000401" not in (run / "evaluation.json").read_text(encoding="utf-8")
    (cache_manifest,) = (tmp_path / "data/processed").glob("*/*/manifest.json")
    assert read_json(cache_manifest)["steps"] == [1, 39]
    with pytest.raises(DatasetError, match="outside protected"):
        evaluate_legacy(training, model_input, scores / "nested", score_run=scores)
