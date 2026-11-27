"""Explain fitted pipelines in their native positive-class output units."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .schema import DatasetError

try:
    from lightgbm import LGBMClassifier
except (ImportError, OSError):
    LGBMClassifier = None


def explain_scores(
    model: Pipeline,
    features: pd.DataFrame,
    scores: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Return signed SHAP contributions, named base values and score residual."""
    classifier = model.named_steps["classifier"]
    probability_units = isinstance(
        classifier, (RandomForestClassifier, ExtraTreesClassifier)
    )
    if (
        isinstance(classifier, XGBClassifier)
        or probability_units
        or (LGBMClassifier is not None and isinstance(classifier, LGBMClassifier))
    ) and len(model.steps) == 1:
        # For sklearn forests, the trees' raw output is the class probability.
        explainer = shap.TreeExplainer(
            classifier, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        result = explainer(features)
    elif isinstance(classifier, LogisticRegression):
        scaler = model.named_steps.get("scale")
        if (
            len(model.steps) != 2
            or not isinstance(scaler, StandardScaler)
            or not scaler.with_mean
        ):
            raise DatasetError(
                "Linear explanations require a fitted, mean-centering StandardScaler."
            )
        # The fitted scaler maps the labelled training feature means to zero.
        reference = np.zeros((1, features.shape[1]))
        explainer = shap.LinearExplainer(classifier, reference)
        result = explainer(scaler.transform(features))
    else:
        raise DatasetError("Unsupported fitted pipeline for explanations.")

    values, base_values = result.values, result.base_values
    if probability_units:
        positive = list(model.classes_).index(1)
        values, base_values = values[:, :, positive], base_values[:, positive]
    contributions = pd.DataFrame(
        values, index=features.index, columns=features.columns, dtype="float64"
    )
    base = pd.Series(
        base_values,
        index=features.index,
        dtype="float64",
        name="base_probability" if probability_units else "base_log_odds",
    )
    margin = base.to_numpy() + contributions.to_numpy().sum(axis=1)
    reconstructed = margin if probability_units else np.exp(-np.logaddexp(0, -margin))
    residual = float(
        np.max(np.abs(reconstructed - scores.loc[features.index].to_numpy()))
    )
    if not np.isfinite(residual) or residual > 2e-6:
        raise DatasetError(
            "SHAP contributions do not reconstruct the saved model's scores."
        )
    return contributions, base, residual


def plot_importance(
    importance: pd.Series, path: Path, *, units: str = "log-odds"
) -> None:
    """Plot aggregate mean absolute SHAP values across all scored transactions."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ordered = importance.sort_values()
    figure = Figure(figsize=(7, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.barh(
        [name.replace("_", " ") for name in ordered.index], ordered, color="#006b86"
    )
    ax.set(
        xlabel=f"Mean absolute SHAP value ({units})",
        title="Feature contributions across the scored batch",
    )
    figure.savefig(path, dpi=150)
