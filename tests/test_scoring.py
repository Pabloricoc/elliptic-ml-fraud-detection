import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from elliptic_fraud import train
from elliptic_fraud.ingestion import load_scoring_graph
from elliptic_fraud.model_features import FEATURE_AUDIT, scoring_features
from elliptic_fraud.schema import CLASSES_FILE, DatasetError
from elliptic_fraud.scoring import score_batch


@pytest.mark.parametrize("periods", [tuple(range(1, 50))], indirect=True)
@pytest.mark.parametrize("winner", ["logistic_regression", "xgboost"])
def test_label_free_cli_reproduces_scores_queue_and_shap_for_either_winner(
    winner: str, model_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise each supported saved pipeline on the same invented graphs.
    candidates = train._candidates()
    monkeypatch.setattr(train, "_candidates", lambda: {winner: candidates[winner]})
    training = train.run_training(model_input, tmp_path / "training", budget_per_step=2)
    classes = pd.read_csv(model_input / CLASSES_FILE, dtype=str).set_index("txId")[
        "class"
    ]
    (model_input / CLASSES_FILE).unlink()
    command = shutil.which("elliptic-score", path=sysconfig.get_path("scripts"))
    assert command is not None
    output = tmp_path / "private scores"
    call = subprocess.run(
        [
            command,
            "--data-dir",
            str(model_input),
            "--model-run",
            str(training),
            "--output-dir",
            str(output),
            "--first-step",
            "40",
            "--last-step",
            "42",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert call.returncode == 0, call.stderr
    (run,) = output.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    scores = pd.read_csv(
        run / "scores.csv", dtype={"tx_id": str}, float_precision="round_trip"
    ).set_index("tx_id")
    queue = pd.read_csv(
        run / "review_queue.csv", dtype={"tx_id": str}, float_precision="round_trip"
    ).set_index("tx_id")
    assert str(run) in call.stdout and scores.index[0] not in call.stdout + call.stderr
    assert list(scores.columns) == [
        "time_step",
        "risk_score",
        "review_rank",
        "above_threshold",
        "in_review_queue",
    ]
    assert set(scores["time_step"]) == {40, 41, 42}
    assert (
        summary["scored"] == len(scores) == 12 and summary["queued"] == len(queue) == 6
    )
    assert (
        summary["model"] == winner
        and classes.loc[scores.index].eq("unknown").sum() == 3
    )
    assert (
        scores.groupby("time_step")["review_rank"].apply(list).tolist()
        == [[1, 2, 3, 4]] * 3
    )
    assert scores.index[scores.in_review_queue].tolist() == queue.index.tolist()
    assert np.array_equal(
        scores.above_threshold, scores.risk_score.ge(summary["threshold"])
    )
    assert "class" not in queue.columns
    model = joblib.load(training / "model.joblib")
    features, _ = scoring_features(load_scoring_graph(model_input), 40, 42)
    expected = train.predict_scores(model, features)
    np.testing.assert_array_equal(scores.risk_score, expected.loc[scores.index])
    shap_columns = [f"shap_{name}" for name in FEATURE_AUDIT]
    margin = queue.base_log_odds + queue[shap_columns].sum(axis=1)
    np.testing.assert_allclose(
        np.exp(-np.logaddexp(0, -margin)), queue.risk_score, atol=2e-6, rtol=0
    )
    assert max(summary["mean_abs_shap"].values()) > 0
    assert set(summary["mean_abs_shap"]) == set(FEATURE_AUDIT)
    if winner == "logistic_regression":
        linear = (
            model.named_steps["scale"].transform(features)
            * model.named_steps["classifier"].coef_[0]
        )
        np.testing.assert_allclose(
            list(pd.Series(summary["mean_abs_shap"]).reindex(FEATURE_AUDIT)),
            np.abs(linear).mean(axis=0),
        )
    reference = pd.read_csv(training / "test_by_step.csv").set_index("time_step")
    outcomes = queue.assign(outcome=classes.loc[queue.index])
    for step, group in outcomes.groupby("time_step"):
        assert (group["outcome"] == "1").sum() == reference.loc[
            step, "confirmed_illicit"
        ]
        assert (group["outcome"] == "unknown").sum() == reference.loc[
            step, "reviewed_unknown"
        ]
    assert (run / "shap_importance.png").read_bytes().startswith(b"\x89PNG")
    assert (run / ".gitignore").read_text().strip() == "*"


def test_scoring_rejects_bad_window_schema_and_corrupted_model_before_deserializing(
    model_input: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = train.run_training(model_input, tmp_path / "training")
    output = tmp_path / "scores"
    with pytest.raises(DatasetError, match="outside"):
        score_batch(training, model_input, model_input)
    with pytest.raises(DatasetError, match="window"):
        score_batch(training, model_input, output, first_step=50)
    manifest_path = training / "manifest.json"
    original = manifest_path.read_bytes()
    bad = json.loads(original)
    bad["features_in_order"] = list(reversed(bad["features_in_order"]))
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(DatasetError, match="feature order"):
        score_batch(training, model_input, output)
    manifest_path.write_bytes(original)
    (training / "model.joblib").write_bytes(b"truncated")
    monkeypatch.setattr(
        joblib, "load", lambda *_: pytest.fail("corrupt artifact was deserialized")
    )
    with pytest.raises(DatasetError, match="SHA-256"):
        score_batch(training, model_input, output)
    assert not output.exists()


def test_scoring_failure_leaves_only_previous_completed_runs(
    model_input: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = train.run_training(model_input, tmp_path / "training")
    output = tmp_path / "scores"
    output.mkdir()
    previous = output / "previous"
    previous.mkdir()

    def fail(*_args: object) -> None:
        raise OSError("unwritable plot")

    monkeypatch.setattr("elliptic_fraud.scoring.plot_importance", fail)
    with pytest.raises(OSError, match="unwritable"):
        score_batch(training, model_input, output)
    assert list(output.iterdir()) == [previous]
