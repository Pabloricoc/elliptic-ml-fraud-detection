"""Measure within-step rank diversity and a three-family ensemble."""

import statistics
from itertools import combinations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .ablation import FAMILY_PARAMS
from .folds import development_folds
from .inference import predict_scores
from .schema import DatasetError
from .tuned_models import ranks_within_step
from .tuning_runs import BoosterArm, TuningRun

MAX_RANK_CORRELATION = 0.97


def rank_correlations(
    predictions: list[pd.DataFrame], time_steps: list[pd.Series]
) -> dict[str, float | None]:
    """Average pairwise Spearman correlations over defined validation steps."""
    ranked = [
        ranks_within_step(scores, steps)
        for scores, steps in zip(predictions, time_steps, strict=True)
    ]
    results = {}
    for left, right in combinations(predictions[0].columns, 2):
        correlations = [
            float(value)
            for frame, steps in zip(ranked, time_steps, strict=True)
            for value in (
                step[left].corr(step[right])
                for _, step in frame.groupby(steps, sort=False)
                if step[left].nunique() > 1 and step[right].nunique() > 1
            )
        ]
        results[f"{left} / {right}"] = (
            statistics.mean(correlations) if correlations else None
        )
    return results


def _redundant(correlations: dict[str, float | None]) -> bool:
    return any(
        value is None or value > MAX_RANK_CORRELATION for value in correlations.values()
    )


def rank_ensemble(
    run: TuningRun,
    xgb: BoosterArm,
    lgbm: BoosterArm,
    forest_reference: dict,
) -> tuple[BoosterArm | None, dict[str, float | None]]:
    """Reject highly correlated members before evaluating the averaged ranks."""
    scenario = xgb.scenario
    folds = development_folds()
    steps = run.dataset.transactions.set_index("tx_id")["time_step"]
    frames = [
        pd.DataFrame({"xgboost": left.scores, "lightgbm": right.scores})
        for left, right in zip(xgb.folds, lgbm.folds, strict=True)
    ]
    if any(frame.isna().any(axis=None) for frame in frames):
        raise DatasetError("The two boosters have different validation rows.")
    aligned_steps = [steps.loc[frame.index] for frame in frames]
    correlations = rank_correlations(frames, aligned_steps)
    if _redundant(correlations):
        return None, correlations

    window = run.window(scenario, forest_reference["features"])
    forest_params = FAMILY_PARAMS["random_forest"]
    forest_config = {
        "scenario": scenario,
        "feature_set": forest_reference["features"],
        "family": "random_forest",
        "params": forest_params,
        "stage": "phase3_rank_member",
        "prediction_jobs": 1,
    }

    def forest_scores(train, validation):
        target = train.known_target()
        model = RandomForestClassifier(**forest_params).fit(
            train.features.loc[target.index], target
        )
        model.set_params(n_jobs=1)
        return predict_scores(model, validation.features).astype(
            "float64"
        ), forest_params["n_estimators"]

    forests = []
    for fold, frame, expected_ap in zip(
        folds, frames, forest_reference["fold_ap"], strict=True
    ):
        forest = run.fold(window, fold, forest_config, forest_scores)
        if forest.report["validation"][
            "mean_step_ap"
        ] != expected_ap or not frame.index.equals(forest.scores.index):
            raise DatasetError(
                "The development forest replay does not match its Phase 3 reference."
            )
        frame["random_forest"] = forest.scores
        forests.append(forest)
    correlations = rank_correlations(frames, aligned_steps)
    if _redundant(correlations):
        return None, correlations
    tiers = f"{xgb.tier} / {lgbm.tier} / {forest_reference['features']}"
    rows = []
    for fold, frame, left, right, forest in zip(
        folds, frames, xgb.folds, lgbm.folds, forests, strict=True
    ):
        rank_scores = ranks_within_step(frame, steps.loc[frame.index]).mean(axis=1)
        config = {
            "scenario": scenario,
            "feature_set": tiers,
            "family": "rank_ensemble",
            "params": {
                "n_estimators": sum(
                    result.report["best_rounds"] for result in (left, right, forest)
                )
            },
            "predictor_count": len(
                {
                    column
                    for member in (left, right, forest)
                    for column in member.report["columns"]
                }
            ),
            "member_digests": [
                member.report["config_sha256"] for member in (left, right, forest)
            ],
            "stage": "rank_average",
        }
        rows.append(
            run.fold(
                window,
                fold,
                config,
                lambda *_, rank_scores=rank_scores: (rank_scores, None),
            )
        )
    return BoosterArm(
        scenario, tiers, "rank_ensemble", "rank average", {}, rows
    ), correlations
