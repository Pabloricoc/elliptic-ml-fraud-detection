"""Load a frozen scoring pipeline and apply its positive-class decision rule."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .model_features import FEATURE_AUDIT
from .schema import DatasetError


@dataclass(frozen=True)
class FrozenModel:
    pipeline: Pipeline
    name: str
    sha256: str
    threshold: float


def load_frozen_model(model_run: Path, manifest: dict) -> FrozenModel:
    """Verify and load a trusted local training artifact with its saved threshold."""
    schema = manifest.get("features_in_order")
    if not isinstance(schema, dict) or list(schema) != list(FEATURE_AUDIT):
        raise DatasetError("Training manifest has an incompatible feature order.")
    threshold = manifest.get("threshold")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise DatasetError("Invalid operating threshold in the training manifest.")

    name = manifest.get("winner")
    expected = {"xgboost": XGBClassifier, "logistic_regression": LogisticRegression}
    if not isinstance(name, str) or name not in expected:
        raise DatasetError("Training manifest specifies an unsupported model.")
    with (model_run / "model.joblib").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest.get("model_sha256"):
            raise DatasetError("Model SHA-256 differs from the training manifest.")
        stream.seek(0)
        model = joblib.load(stream)
    if (
        not isinstance(model, Pipeline)
        or list(getattr(model, "feature_names_in_", [])) != list(FEATURE_AUDIT)
        or list(getattr(model, "classes_", [])) != [0, 1]
        or not isinstance(model.named_steps.get("classifier"), expected[name])
    ):
        raise DatasetError("Incompatible fitted model, predictors or class order.")
    return FrozenModel(model, name, digest, float(threshold))


def predict_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return the illicit-class score for each transaction, preserving its join key."""
    positive = list(model.classes_).index(1)
    scores = pd.Series(
        model.predict_proba(features)[:, positive], index=features.index, name="score"
    )
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise DatasetError("Model returned an invalid positive-class score.")
    return scores
