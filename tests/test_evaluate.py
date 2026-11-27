import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from elliptic_fraud import evaluate, experiments
from elliptic_fraud.evaluate import evaluate_legacy
from elliptic_fraud.model_features import FEATURE_AUDIT, WINDOWS
from elliptic_fraud.runs import file_digest, read_json, write_json
from elliptic_fraud.schema import EXPECTED_FILES, DatasetError
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


def test_development_cache_failure_preserves_published_evaluation(
    model_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = run_training(model_input, tmp_path / "trained")
    output = tmp_path / "evaluations"
    original_ap = read_json(training / "metrics.json")["test"]["average_precision"]

    def fail_cache(*args: object) -> None:
        (published,) = output.glob("evaluation-*/evaluation.json")
        assert read_json(published)["test"]["pooled_ap"] == original_ap
        raise OSError("simulated cache write failure")

    monkeypatch.setattr(experiments, "cache_development", fail_cache)
    with pytest.raises(
        DatasetError, match="Evaluation was published.*cache write failure"
    ):
        evaluate_legacy(training, model_input, output, prepare_development=True)
    (published,) = output.glob("evaluation-*/evaluation.json")
    assert read_json(published)["test"]["pooled_ap"] == original_ap


@pytest.mark.parametrize(
    "field",
    [
        ("manifest.json", "label_policy"),
        ("manifest.json", "availability"),
        ("manifest.json", "model_sha256"),
        ("manifest.json", "windows_inclusive", "test"),
        ("manifest.json", "dataset", "files_sha256", EXPECTED_FILES[0]),
        ("manifest.json", "dataset", "snapshot_sha256"),
        ("metrics.json", "test"),
        ("metrics.json", "test", "average_precision"),
        ("metrics.json", "test", "operating_point"),
        ("metrics.json", "test", "operating_point", "tp"),
        ("metrics.json", "test", "review_queue"),
        ("metrics.json", "test", "review_queue", "confirmed_illicit"),
    ],
)
def test_missing_legacy_metadata_fails_before_accessing_inputs(
    field: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = tmp_path / "trained"
    training.mkdir()
    metadata = {
        "manifest.json": {
            "dataset": {
                "files_sha256": dict.fromkeys(EXPECTED_FILES, "0" * 64),
                "snapshot_sha256": "0" * 64,
            },
            "windows_inclusive": WINDOWS.copy(),
            "seed": 42,
            "review_budget_per_step": 50,
            "features_in_order": FEATURE_AUDIT,
            "winner": "xgboost",
            "threshold": 0.5,
            "model_sha256": "0" * 64,
            "label_policy": "known labels only",
            "availability": "confirmation-time ancestry",
        },
        "metrics.json": {
            "test": {
                "average_precision": 0.25,
                "operating_point": dict.fromkeys(
                    ("tp", "fp", "fn", "tn", "precision", "recall", "f1"), 0
                ),
                "review_queue": dict.fromkeys(
                    (
                        "eligible",
                        "labelled",
                        "known_illicit",
                        "reviewed",
                        "confirmed_illicit",
                        "confirmed_licit",
                        "reviewed_unknown",
                        "budget_per_step",
                        "recall_known_illicit",
                    ),
                    0,
                ),
            }
        },
    }
    parent = metadata
    for key in field[:-1]:
        parent = parent[key]
    del parent[field[-1]]
    for filename, document in metadata.items():
        write_json(training / filename, document)

    def unexpected_access(*args: object) -> None:
        pytest.fail("Replay accessed inputs before validating its metadata.")

    for name in ("file_state", "load_dataset", "load_frozen_model", "predict_scores"):
        monkeypatch.setattr(evaluate, name, unexpected_access)
    with pytest.raises(DatasetError, match=f"Missing required field .*{field[-1]}"):
        evaluate_legacy(training, tmp_path / "unused inputs", tmp_path / "evaluations")
