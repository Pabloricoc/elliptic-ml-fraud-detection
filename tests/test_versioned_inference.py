from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from elliptic_fraud.api import create_app
from elliptic_fraud.evaluate import evaluate_legacy
from elliptic_fraud.explanations import explain_scores
from elliptic_fraud.inference import load_frozen_model, predict_scores
from elliptic_fraud.ingestion import EllipticDataset, load_scoring_graph
from elliptic_fraud.model_features import build_feature_set, get_feature_set
from elliptic_fraud.runs import file_digest, read_json, write_json
from elliptic_fraud.schema import CLASSES_FILE, DatasetError
from elliptic_fraud.scoring import score_batch
from elliptic_fraud.train import run_training


@pytest.mark.parametrize("periods", [tuple(range(1, 6))], indirect=True)
@pytest.mark.parametrize(
    ("family", "classifier", "scenario", "name", "units"),
    [
        ("logistic_regression", LogisticRegression(), "S1", "T1", "log-odds"),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=6,
                max_depth=2,
                min_child_weight=0,
                n_jobs=1,
                random_state=42,
            ),
            "S1",
            "T0+T1",
            "log-odds",
        ),
        (
            "lightgbm",
            LGBMClassifier(
                objective="binary",
                n_estimators=6,
                min_child_samples=1,
                deterministic=True,
                n_jobs=1,
                random_state=42,
                verbosity=-1,
            ),
            "S1",
            "T1",
            "log-odds",
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=7, random_state=42),
            "S2",
            "T1b+T2+T3",
            "probability",
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(n_estimators=7, random_state=42),
            "S2",
            "T2agg+T1b",
            "probability",
        ),
    ],
)
def test_version_two_batch_scores_and_reconstructs_each_model_family(
    family: str,
    classifier: object,
    scenario: str,
    name: str,
    units: str,
    periods: EllipticDataset,
    model_input: Path,
    tmp_path: Path,
) -> None:
    feature_set = get_feature_set(name, scenario)
    graph = load_scoring_graph(model_input, feature_set.required_columns)
    features, _ = build_feature_set(graph, feature_set, 1, 3)
    labels = periods.transactions.set_index("tx_id").loc[features.index, "class"]
    known = labels.isin(["1", "2"])
    preprocessing = (
        [("scale", StandardScaler())] if family == "logistic_regression" else []
    )
    pipeline = Pipeline([*preprocessing, ("classifier", classifier)]).fit(
        features.loc[known], labels[known].eq("1").astype(int)
    )
    model_run = tmp_path / "frozen"
    model_run.mkdir()
    artifact = model_run / "model.joblib"
    joblib.dump(pipeline, artifact)
    manifest = {
        "schema_version": 2,
        "feature_set": feature_set.manifest(),
        "model_family": family,
        "params": {"random_state": 42},
        "fold_protocol": {"train": [1, 3], "validation": [4, 4]},
        "windows_inclusive": {"test": [5, 5]},
        "threshold": 0.5,
        "seed": 42,
        "review_budget_per_step": 4,
        "model_sha256": file_digest(artifact),
    }
    write_json(model_run / "manifest.json", manifest)
    frozen = load_frozen_model(model_run, manifest)
    assert frozen.feature_set is not None
    assert frozen.feature_set.manifest() == manifest["feature_set"]
    assert frozen.name == family

    # Batch builds the registered predictors with no access to outcome labels.
    (model_input / CLASSES_FILE).unlink()
    scored = score_batch(model_run, model_input, tmp_path / "scored")
    report = read_json(scored / "summary.json")
    queue = pd.read_csv(
        scored / "review_queue.csv", dtype={"tx_id": str}, float_precision="round_trip"
    ).set_index("tx_id")
    expected, _ = build_feature_set(graph, feature_set, 5, 5)
    np.testing.assert_array_equal(
        queue["risk_score"], predict_scores(pipeline, expected).loc[queue.index]
    )
    assert report["features"] == list(feature_set.columns)
    assert report["explanation_units"] == units
    assert report.get("availability_assumption") == (
        "conditional on A1" if scenario == "S2" else None
    )
    contributions = queue[[f"shap_{name}" for name in feature_set.columns]].sum(axis=1)
    if units == "probability":
        reconstructed = queue["base_probability"] + contributions
    else:
        margin = queue["base_log_odds"] + contributions
        reconstructed = np.exp(-np.logaddexp(0, -margin))
    np.testing.assert_allclose(reconstructed, queue["risk_score"], atol=2e-6, rtol=0)
    assert max(report["mean_abs_shap"].values()) > 0

    manifest["feature_set"]["columns"] = list(reversed(feature_set.columns))
    with pytest.raises(DatasetError, match="feature order"):
        load_frozen_model(model_run, manifest)


def test_linear_shap_reference_is_the_fitted_training_mean() -> None:
    training = pd.DataFrame(
        {"a": [2.0, 4.0, 14.0, 24.0], "b": [11.0, 12.0, 16.0, 30.0]}
    )
    target = [0, 0, 1, 1]
    model = Pipeline(
        [("scale", StandardScaler()), ("classifier", LogisticRegression())]
    ).fit(training, target)
    mean = training.mean().to_frame().T
    scoring = pd.concat(
        [mean, pd.DataFrame({"a": [120.0], "b": [180.0]})], ignore_index=True
    )
    contributions, base, _ = explain_scores(
        model, scoring, predict_scores(model, scoring)
    )
    np.testing.assert_allclose(
        base, model.decision_function(mean).item(), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(contributions.iloc[0], [0, 0])

    unscaled = Pipeline([("classifier", LogisticRegression())]).fit(training, target)
    with pytest.raises(DatasetError, match="mean-centering StandardScaler"):
        explain_scores(unscaled, scoring, predict_scores(unscaled, scoring))


def test_lightgbm_shap_dispatches_by_fitted_estimator_type() -> None:
    features = pd.DataFrame({"a": np.arange(16, dtype=float)})
    model = Pipeline(
        [
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=4,
                    min_child_samples=1,
                    n_jobs=1,
                    verbosity=-1,
                ),
            )
        ]
    ).fit(features, features["a"].ge(8).astype(int))
    scores = predict_scores(model, features)
    contributions, base, residual = explain_scores(model, features, scores)
    assert base.name == "base_log_odds"
    assert contributions["a"].abs().max() > 0
    assert residual < 2e-6


def test_implicit_and_explicit_v1_load_the_same_frozen_model(
    model_input: Path, tmp_path: Path
) -> None:
    model_run = run_training(model_input, tmp_path / "trained")
    manifest = read_json(model_run / "manifest.json")
    implicit = load_frozen_model(model_run, manifest)
    explicit = load_frozen_model(model_run, {**manifest, "schema_version": 1})
    assert implicit.sha256 == explicit.sha256
    assert implicit.feature_set is explicit.feature_set is None


@pytest.mark.parametrize(
    ("schema", "changes", "message"),
    [
        ({"version": 3}, {}, "feature-set version"),
        ({"name": "T2"}, {}, "requested scenario"),
        ({}, {"model_family": "unsupported"}, "unsupported model"),
        ({}, {"fold_protocol": None}, "fold protocol"),
    ],
)
def test_v2_invalid_contract_is_rejected_without_a_model_file(
    schema: dict, changes: dict, message: str, tmp_path: Path
) -> None:
    manifest = {
        "schema_version": 2,
        "feature_set": {**get_feature_set("T1", "S1").manifest(), **schema},
        "model_family": "logistic_regression",
        "params": {},
        "fold_protocol": {"name": "test-folds"},
        "threshold": 0.5,
        **changes,
    }
    with pytest.raises(DatasetError, match=message):
        load_frozen_model(tmp_path, manifest)


def test_unapproved_v2_contracts_stop_before_legacy_serving_or_replay(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / "manifest.json", {"schema_version": 2})
    with pytest.raises(DatasetError, match="only legacy"):
        with TestClient(create_app(tmp_path)):
            pytest.fail("The legacy API loaded a version 2 artifact.")
    write_json(tmp_path / "metrics.json", {})
    with pytest.raises(DatasetError, match="requires a version 1"):
        evaluate_legacy(tmp_path, tmp_path / "input", tmp_path.parent / "outside")
