"""Fit CPU boosters with an inner, strictly chronological stopping window."""

from math import sqrt

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from optuna import Trial
from xgboost import XGBClassifier

from .folds import Fold
from .inference import predict_scores
from .model_features import Window
from .schema import DatasetError

FAMILIES = ("xgboost", "lightgbm")
SEEDS = (42, 43, 44, 45, 46)
PATIENCE = 100


def suggest_parameters(trial: Trial, family: str) -> dict:
    """Sample the registered family-specific search space."""
    if family == "xgboost":
        values = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 50, log=True
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
    else:
        values = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    values["weight"] = trial.suggest_categorical(
        "weight", ["unit", "sqrt_ratio", "ratio"]
    )
    return values


def _weight(mode: str, target: pd.Series) -> float:
    positives = int(target.sum())
    if not 0 < positives < len(target):
        raise DatasetError("The booster fitting window needs both known classes.")
    ratio = (len(target) - positives) / positives
    return {"unit": 1.0, "sqrt_ratio": sqrt(ratio), "ratio": ratio}[mode]


def booster_parameters(
    family: str, params: dict, seed: int, rounds: int, target: pd.Series
) -> dict:
    """Resolve fixed options and compute the class weight from fitting labels only."""
    output = {
        **{name: value for name, value in params.items() if name != "weight"},
        "random_state": seed,
        "n_estimators": rounds,
        "n_jobs": 6,
        "scale_pos_weight": _weight(params["weight"], target),
    }
    if family == "xgboost":
        return {
            **output,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cpu",
        }
    return {
        **output,
        "objective": "binary",
        "deterministic": True,
        "force_col_wise": True,
        "subsample_freq": 1,
        "verbosity": -1,
    }


def fit_booster(
    family: str,
    params: dict,
    train: Window,
    validation: Window,
    fold: Fold,
    *,
    max_rounds: int,
    seed: int,
    fixed_rounds: int | None = None,
) -> tuple[pd.Series, int]:
    """Use inner log loss to choose rounds, then refit and score the outer fold."""
    if (train.first, train.last) != fold.train or (
        validation.first,
        validation.last,
    ) != fold.validation:
        raise DatasetError("Booster inputs must match the declared temporal fold.")
    target = train.known_target()
    estimator = XGBClassifier if family == "xgboost" else LGBMClassifier
    rounds = fixed_rounds
    if rounds is None:
        inner_mask = train.steps.loc[target.index].between(*fold.inner_early_stopping)
        inner, core = target.loc[inner_mask], target.loc[~inner_mask]
        if inner.empty:
            raise DatasetError("Inner early stopping needs known labels.")
        options = booster_parameters(family, params, seed, max_rounds, core)
        inner_features = train.features.loc[inner.index]
        if family == "xgboost":
            model = estimator(**options, early_stopping_rounds=PATIENCE)
            model.fit(
                train.features.loc[core.index],
                core,
                eval_set=[(inner_features, inner)],
                verbose=False,
            )
            rounds = model.best_iteration + 1
        else:
            model = estimator(**options)
            model.fit(
                train.features.loc[core.index],
                core,
                eval_X=inner_features,
                eval_y=inner,
                eval_metric="binary_logloss",
                callbacks=[early_stopping(PATIENCE, verbose=False)],
            )
            rounds = model.best_iteration_
    if not 1 <= rounds <= max_rounds:
        raise DatasetError(
            "The selected iteration count is outside the configured limit."
        )
    model = estimator(**booster_parameters(family, params, seed, rounds, target))
    model.fit(train.features.loc[target.index], target)
    model.set_params(n_jobs=1)
    return predict_scores(model, validation.features).astype("float64"), rounds


def ranks_within_step(scores: pd.DataFrame, steps: pd.Series) -> pd.DataFrame:
    """Represent every model score as an average-tie within-step percentile."""
    if not scores.index.equals(steps.index):
        raise DatasetError("Ensemble scores and time steps must be aligned.")
    return scores.groupby(steps, sort=False).rank(method="average", pct=True)
