from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from optuna.trial import create_trial
from scripts import replay_phase4_rounds

from elliptic_fraud import tuned_models, tuning_runs, tuning_search
from elliptic_fraud.folds import development_folds, slice_development
from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.model_features import build_window
from elliptic_fraud.runs import snapshot_digest, write_json
from elliptic_fraud.schema import DatasetError
from elliptic_fraud.tuned_models import fit_booster, ranks_within_step
from elliptic_fraud.tuning import TIERS, _limits
from elliptic_fraud.tuning_ensemble import rank_correlations
from elliptic_fraud.tuning_runs import TuningRun
from elliptic_fraud.tuning_search import (
    Search,
    plateau_reached,
    seed_average,
    tune_tier,
)


def test_settings_and_plateau_respect_per_family_budget() -> None:
    assert _limits({"trials_per_tier": 30, "max_rounds": 2000, "tiers": TIERS}) == (
        30,
        2000,
    )
    with pytest.raises(DatasetError, match="at most 60"):
        _limits({"trials_per_tier": 31, "max_rounds": 2000, "tiers": TIERS})
    with pytest.raises(DatasetError, match="approved"):
        _limits(
            {"trials_per_tier": 4, "max_rounds": 300, "tiers": {**TIERS, "S1": ["T2"]}}
        )
    trials = [create_trial(value=0.21), *[create_trial(value=0.211) for _ in range(30)]]
    assert not plateau_reached(trials[:-1])
    assert plateau_reached(trials)
    assert not plateau_reached([*trials[:-1], create_trial(value=0.214)])


@pytest.mark.parametrize("periods", [tuple(range(1, 40))], indirect=True)
def test_early_stopping_uses_inner_labels_and_refits_before_outer_prediction(
    periods: EllipticDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    class RecordingXGBoost:
        classes_ = [0, 1]
        best_iteration = 1

        def __init__(self, **options) -> None:
            self.options = options

        def fit(self, x, y, eval_set=None, **kwargs):
            events.append((x.index, y, self.options, eval_set))
            return self

        def set_params(self, **kwargs):
            return self

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5)

    monkeypatch.setattr(tuned_models, "XGBClassifier", RecordingXGBoost)
    development = build_window(periods, 1, 39)
    fold = development_folds(2)[0]
    train = slice_development(development, fold.train)
    validation = slice_development(development, fold.validation)
    scores, rounds = fit_booster(
        "xgboost", {"weight": "ratio"}, train, validation, fold, max_rounds=4, seed=42
    )
    core, final = events
    assert core[2]["early_stopping_rounds"] == 100
    assert core[2]["scale_pos_weight"] == 2
    assert core[2]["n_jobs"] == 6
    assert development.steps.loc[core[0]].max() == 14
    inner_x, inner_y = core[3][0]
    assert set(development.steps.loc[inner_x.index]) == {15, 16, 17}
    assert "unknown" not in set(development.labels.loc[inner_y.index])
    assert final[3] is None and rounds == final[2]["n_estimators"] == 2
    assert development.steps.loc[final[0]].max() == 17
    assert "unknown" not in set(development.labels.loc[final[0]])
    assert scores.index.equals(validation.steps.index)


@pytest.mark.parametrize("periods", [tuple(range(1, 40))], indirect=True)
@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_search_resume_and_five_seeds_use_cached_development_folds(
    family: str,
    periods: EllipticDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tuning_runs, "record_fold_mlflow", lambda _, row: row["config_sha256"]
    )
    window = build_window(periods, 1, 39)
    run = TuningRun(
        periods, tmp_path / "features", tmp_path, {"test": "synthetic"}, "synthetic"
    )
    search = Search("S1", "T0", family, 1, 2)
    storage = tmp_path / "study.db"
    arms, best, trials = tune_tier(run, storage, window, search, [])
    assert len(arms) == len(trials) == 1
    assert len(best.folds) == 4 and trials[0].state.name == "COMPLETE"
    mean = seed_average(run, window, search, best, development_folds())
    assert all(len(row.report["config"]["members"]) == 5 for row in mean.folds)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("The completed search or seed fit was repeated.")

    monkeypatch.setattr(tuning_search, "fit_booster", unexpected_fit)
    _, replay, _ = tune_tier(run, storage, window, search, [])
    assert replay.aggregate() == best.aggregate()
    assert (
        seed_average(run, window, search, replay, development_folds()).aggregate()
        == mean.aggregate()
    )


def test_ranks_keep_ties_and_compute_correlations_within_steps() -> None:
    scores = pd.DataFrame({"a": [1, 1, 2, 0, 1, 2], "b": [5, 5, 9, 8, 4, 0]})
    steps = pd.Series([1, 1, 1, 2, 2, 2])
    ranks = ranks_within_step(scores, steps)
    assert list(ranks["a"]) == [0.5, 0.5, 1, 1 / 3, 2 / 3, 1]
    assert rank_correlations([scores], [steps])["a / b"] == pytest.approx(0.0)


def test_frozen_replay_rejects_a_different_fold_schedule(tmp_path: Path) -> None:
    def archived_arm(stage: str, embargo: int) -> dict:
        params = {"weight": "unit", "max_depth": 3}
        digests = []
        for fold in development_folds(embargo):
            config = {
                "fold": asdict(fold),
                "params": {**params, "n_estimators": 300},
                "stage": stage,
            }
            digest = snapshot_digest(config)
            folder = tmp_path / digest
            folder.mkdir()
            write_json(folder / "evaluation.json", {"config": config})
            digests.append(digest)
        return {
            "id": snapshot_digest({"folds": digests}),
            "scenario": "S2",
            "family": "xgboost",
            "features": "T1b+T2+T3",
            "stage": stage,
            "search_parameters": params,
            "folds": [{"config_sha256": digest} for digest in digests],
        }

    base = archived_arm("TPE", 0)
    nominal = archived_arm("5 seeds", 0)
    embargoed = archived_arm("5 seeds", 2)
    archive = {"arms": [base, nominal, embargoed]}
    assert replay_phase4_rounds._frozen_pair(
        archive, tmp_path, (base["id"][:10], nominal["id"][:10])
    ) == (base, nominal)
    with pytest.raises(DatasetError, match="does not match the frozen replay"):
        replay_phase4_rounds._frozen_pair(
            archive, tmp_path, (base["id"][:10], embargoed["id"][:10])
        )
