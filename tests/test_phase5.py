from pathlib import Path
from statistics import mean

import numpy as np
import pytest
from xgboost import XGBClassifier

from elliptic_fraud import phase5
from elliptic_fraud.development import export_development, load_development_export
from elliptic_fraud.folds import slice_development
from elliptic_fraud.inference import FrozenModel
from elliptic_fraud.phase5_models import (
    ARM_NAMES,
    S1_TIER,
    S2_TIER,
    Arm,
    _fit,
    _training_window,
    freeze_models,
    load_arms,
)
from elliptic_fraud.phase5_report import paired_contrasts, step_draws
from elliptic_fraud.runs import file_digest, read_json
from elliptic_fraud.schema import DatasetError
from elliptic_fraud.train import BOOST_PARAMS


def test_paired_steps_preserve_missing_values_and_registered_directions() -> None:
    scores = {
        "s2_lgbm": {40: 0.6, 41: 0.4, 42: None},
        "s2_rf": {40: 0.2, 41: 0.3, 42: None},
        "s1_legacy": {40: 0.8, 41: 0.7, 42: None},
        "s1_refit": {40: 0.3, 41: 0.7, 42: None},
    }
    indices = np.array([[0, 0, 2], [1, 1, 2], [2, 2, 2], [0, 1, 2]])
    result = paired_contrasts(
        {k: {"per_step_ap": v} for k, v in scores.items()}, indices
    )
    assert [row["steps"] for row in result] == [2, 2, 2]
    assert [row["defined_draws"] for row in result] == [3, 3, 3]
    assert result[0]["estimate"] == pytest.approx(mean([0.4, 0.1]))
    assert "has an advantage" in result[0]["conclusion"]
    assert "has a disadvantage" in result[1]["conclusion"]
    assert "includes zero" in result[2]["conclusion"]
    assert result[2]["role"] == "Descriptive"
    np.testing.assert_array_equal(step_draws(10), step_draws(10))


@pytest.mark.parametrize("periods", [tuple(range(1, 50))], indirect=True)
def test_frozen_four_arm_evaluation_publishes_outputs_and_rejects_retry(
    model_input: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "development"
    export_development(model_input, directory)
    dataset, manifest = load_development_export(directory)
    window = _training_window(dataset, S1_TIER)
    pipeline = _fit(XGBClassifier(**BOOST_PARAMS), slice_development(window, (1, 29)))
    legacy = FrozenModel(pipeline, "xgboost", "synthetic", 0.5)
    root = tmp_path / "experiment"
    provenance = {
        "git_commit": "synthetic",
        "approval_commit": "synthetic",
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "environment": {"synthetic": True},
    }
    frozen = freeze_models(dataset, root, provenance, legacy)
    freeze = read_json(frozen / "freeze.json")
    sha = file_digest(frozen / "freeze.json")
    assert freeze["s1_refit_threshold"]["train"] == [1, 34]
    assert freeze["s1_refit_threshold"]["validation"] == [35, 39]
    arms = load_arms(frozen, freeze, legacy)
    assert tuple(len(arm.members) for arm in arms.values()) == (1, 1, 1, 5)
    result = phase5.evaluate_once(root, model_input, freeze, arms, freeze_sha=sha)
    receipt = read_json(root / "holdout-attempt.json")
    report = read_json(result / "evaluation.json")
    assert receipt["status"] == "completed"
    assert receipt["evaluation_sha256"] == file_digest(result / "evaluation.json")
    assert tuple(report["arms"]) == tuple(ARM_NAMES)
    assert all(arm["defined_steps"] == 10 for arm in report["arms"].values())
    assert len(report["contrasts"]) == 3
    assert all(contrast["defined_draws"] == 1000 for contrast in report["contrasts"])
    assert "## Per-step Q-all outcomes" in (result / "evaluation.md").read_text(
        encoding="utf-8"
    )
    assert (result / "s1_legacy" / "scores.csv").is_file()
    assert (result / "s2_lgbm" / "review_queue.csv").is_file()
    with pytest.raises(DatasetError, match="attempt already exists"):
        phase5.evaluate_once(root, model_input, freeze, arms, freeze_sha=sha)


def test_holdout_ingestion_failure_preserves_attempt_and_prevents_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fail(raw: Path, snapshot: str) -> None:
        calls.append((raw, snapshot))
        receipt = read_json(tmp_path / "holdout-attempt.json")
        assert receipt["status"] == "started"
        raise DatasetError("Synthetic ingestion error")

    monkeypatch.setattr(phase5, "_load_test", fail)
    arms = {
        key: Arm((), S1_TIER if key.startswith("s1") else S2_TIER, 0.5)
        for key in ARM_NAMES
    }
    freeze = {
        "git_commit": "synthetic",
        "approval_commit": "synthetic",
        "source_snapshot_sha256": "synthetic",
        "arms": {},
    }
    with pytest.raises(DatasetError, match="Do not retry"):
        phase5.evaluate_once(tmp_path, tmp_path, freeze, arms, freeze_sha="synthetic")
    receipt = read_json(tmp_path / "holdout-attempt.json")
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "release verification and ingestion"
    with pytest.raises(DatasetError, match="attempt already exists"):
        phase5.evaluate_once(tmp_path, tmp_path, freeze, arms, freeze_sha="synthetic")
    assert len(calls) == 1
