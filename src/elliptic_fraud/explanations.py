"""Explain the frozen Phase 2 pipelines on their positive-class log-odds scale."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .schema import DatasetError


def explain_scores(
    model: Pipeline, features: pd.DataFrame, scores: pd.Series
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Return signed SHAP contributions, base log-odds and maximum score residual."""
    classifier = model.named_steps["classifier"]
    if isinstance(classifier, XGBClassifier) and len(model.steps) == 1:
        explainer = shap.TreeExplainer(
            classifier, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        result = explainer(features)
    elif (
        isinstance(classifier, LogisticRegression)
        and isinstance(model.named_steps.get("scale"), StandardScaler)
        and len(model.steps) == 2
    ):
        scaler = model.named_steps["scale"]
        # Zero in scaled space is the labelled training mean fitted by StandardScaler.
        reference = np.zeros((1, features.shape[1]))
        explainer = shap.LinearExplainer(classifier, reference)
        result = explainer(scaler.transform(features))
    else:
        raise DatasetError("Unsupported fitted pipeline for log-odds explanations.")

    contributions = pd.DataFrame(
        result.values, index=features.index, columns=features.columns, dtype="float64"
    )
    base = pd.Series(
        result.base_values, index=features.index, dtype="float64", name="base_log_odds"
    )
    margin = base.to_numpy() + contributions.to_numpy().sum(axis=1)
    reconstructed = np.exp(-np.logaddexp(0, -margin))
    residual = float(
        np.max(np.abs(reconstructed - scores.loc[features.index].to_numpy()))
    )
    if not np.isfinite(residual) or residual > 2e-6:
        raise DatasetError(
            "SHAP contributions do not reconstruct the saved model's scores."
        )
    return contributions, base, residual


def plot_importance(importance: pd.Series, path: Path) -> None:
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
        xlabel="Mean absolute SHAP value (log-odds)",
        title="Feature contributions across the scored batch",
    )
    figure.savefig(path, dpi=150)
