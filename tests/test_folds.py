from dataclasses import replace
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from elliptic_fraud.experiments import (
    TRACKING_EXPERIMENT,
    cache_development,
    run_baseline_folds,
)
from elliptic_fraud.folds import development_folds, slice_development
from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.model_features import build_window
from elliptic_fraud.runs import read_json
from elliptic_fraud.schema import DatasetError


def test_fixed_expanding_folds_and_optional_embargo() -> None:
    folds = development_folds()
    assert [(fold.train, fold.validation) for fold in folds] == [
        ((1, 19), (20, 24)),
        ((1, 24), (25, 29)),
        ((1, 29), (30, 34)),
        ((1, 34), (35, 39)),
    ]
    for plain, embargoed in zip(folds, development_folds(2), strict=True):
        assert embargoed.train == (1, plain.train[1] - 2)
        assert embargoed.validation == plain.validation
        assert embargoed.inner_early_stopping == (
            plain.train[1] - 4,
            plain.train[1] - 2,
        )
    with pytest.raises(ValueError, match="embargo|Embargo"):
        development_folds(-1)
    with pytest.raises(ValueError, match="inner training split"):
        development_folds(19)


@pytest.mark.parametrize("periods", [tuple(range(1, 50))], indirect=True)
def test_walk_forward_fits_only_development_and_deduplicates_runs(
    periods: EllipticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    development = build_window(periods, 1, 39)
    contaminated = development.steps.copy()
    contaminated.iloc[-1] = 40
    with pytest.raises(DatasetError, match="limited to steps"):
        cache_development(
            replace(development, steps=contaminated), {"snapshot_sha256": "a" * 64}
        )
    with pytest.raises(DatasetError, match="outside"):
        slice_development(development, (38, 40))
    cache = cache_development(development, {"snapshot_sha256": "a" * 64})
    output = tmp_path / "experiments"
    result = run_baseline_folds(cache, output, git_commit="baseline-test")
    summary = read_json(result)
    assert summary["embargo"] == 0
    assert len(summary["folds"]) == 8
    assert {row["model"] for row in summary["folds"]} == {
        "logistic_regression",
        "xgboost",
    }
    for row in summary["folds"]:
        fold = next(fold for fold in development_folds() if fold.name == row["fold"])
        report = read_json(output / row["config_sha256"] / "evaluation.json")
        assert report["train"]["last_step"] == fold.train[1]
        assert report["validation"]["support"]["last_step"] == fold.validation[1]
        assert report["validation"]["support"]["unknown"] == 5
        assert report["git_commit"] == "baseline-test"

    def unexpected_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("A duplicate experiment should reuse its saved fit report.")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    assert run_baseline_folds(cache, output, git_commit="later-revision") == result
    from mlflow import MlflowClient

    client = MlflowClient(
        tracking_uri=f"sqlite:///{(output / 'tracking/mlflow.db').as_posix()}"
    )
    experiment = client.get_experiment_by_name(TRACKING_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert {run.info.run_id for run in runs} == {
        row["mlflow_run_id"] for row in summary["folds"]
    }
    assert all(run.info.status == "FINISHED" for run in runs)
