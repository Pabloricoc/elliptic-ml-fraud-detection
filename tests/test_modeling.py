import hashlib
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from elliptic_fraud.ingestion import EllipticDataset, load_dataset
from elliptic_fraud.model_evaluation import (
    at_threshold,
    average_precision,
    review_queue,
    validation_threshold,
)
from elliptic_fraud.model_features import (
    FEATURE_AUDIT,
    WINDOWS,
    ancestry_features,
    build_window,
)
from elliptic_fraud.schema import CLASSES_FILE, FEATURES_FILE, DatasetError
from elliptic_fraud.train import fit_and_select, main, predict_scores


def test_whole_steps_ancestry_and_label_isolation(periods: EllipticDataset) -> None:
    windows = [build_window(periods, *bounds) for bounds in WINDOWS.values()]
    assert [sorted(w.steps.unique()) for w in windows] == [[1, 29], [30, 39], [40, 49]]
    assert len(set.union(*(set(w.features.index) for w in windows))) == len(
        periods.transactions
    )
    a, b, c, d = periods.transactions["tx_id"].iloc[:4]
    train = windows[0]
    assert list(train.features.columns) == list(FEATURE_AUDIT)
    assert train.features.loc[[a, b, c, d]].values.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [2, 0.5, 1, 2],
        [0, 0, 0, 0],
    ]
    assert train.known_target().loc[[b, c, d]].tolist() == [0, 1, 0]
    assert a not in train.known_target().index and a in train.features.index

    later = "000000019"
    changed = periods.transactions.assign(feature_001=999, **{"class": "unknown"})
    changed.loc[len(changed)] = [later, 1, "unknown", -999]
    edges = pd.concat(
        [
            periods.edges,
            pd.DataFrame([(c, later), (a, b)], columns=periods.edges.columns),
        ]
    )
    observed = build_window(EllipticDataset(changed, edges), 1, 29)
    pd.testing.assert_frame_equal(
        train.features, observed.features.loc[train.features.index]
    )


def test_unresolved_inputs_and_directed_cycles_reject_ancestry(
    periods: EllipticDataset,
) -> None:
    ids = pd.Index(["a", "b"])
    edges = pd.DataFrame([("a", "b"), ("b", "a")], columns=["source_id", "target_id"])
    with pytest.raises(DatasetError, match="cycle"):
        ancestry_features(ids, edges)
    with pytest.raises(DatasetError, match="outside"):
        ancestry_features(pd.Index(["a"]), edges)
    with pytest.raises(DatasetError, match="No eligible"):
        build_window(periods, 2, 3)


def test_training_only_scaler_and_validation_tie_rule(
    periods: EllipticDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, valid = (
        build_window(periods, *WINDOWS[key]) for key in ("train", "validation")
    )
    candidates = {
        name: Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", DummyClassifier(strategy="prior")),
            ]
        )
        for name in ("logistic_regression", "xgboost")
    }
    monkeypatch.setattr("elliptic_fraud.train._candidates", lambda: candidates)
    valid.features += 1000
    selected = fit_and_select(train, valid)
    assert selected.name == "logistic_regression"
    assert len(set(selected.candidate_ap.values())) == 1
    expected = train.features.loc[train.known_target().index]
    for model in candidates.values():
        scale = model.named_steps["scale"]
        assert scale.n_samples_seen_ == len(expected)
        np.testing.assert_allclose(scale.mean_, expected.mean())
    with pytest.raises(DatasetError, match="precede"):
        fit_and_select(valid, train)
    valid.labels = valid.labels.replace("1", "unknown")
    with pytest.raises(DatasetError, match="require labelled"):
        fit_and_select(train, valid)


def test_threshold_ties_and_scores_align_by_id() -> None:
    y = pd.Series([1, 0, 0, 1], index=list("abcd"))
    score = pd.Series([0.1, 0.3, 0.5, 0.9, 0.99], index=list("abcdu")).sort_index(
        ascending=False
    )
    assert validation_threshold(y, score) == 0.9
    assert average_precision(y, score) == pytest.approx((1.0 + 0.5) / 2)
    metrics = at_threshold(y, score, 0.9)
    assert metrics == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "tn": 2,
        "precision": 1,
        "recall": 0.5,
        "f1": 2 / 3,
    }
    assert average_precision(y.iloc[1:3], score) is None
    assert at_threshold(y.iloc[:0], score, 0.9)["recall"] is None
    assert at_threshold(y, score, 1.0)["precision"] is None


def test_queue_ranks_unknowns_and_handles_empty_steps_and_ties(
    periods: EllipticDataset,
) -> None:
    window = build_window(periods, *WINDOWS["test"])
    scores = window.labels.map({"unknown": 1.0, "1": 0.5, "2": 0.5})
    totals, steps = review_queue(window, scores, budget=1, seed=42)
    assert totals["reviewed"] == totals["reviewed_unknown"] == 2
    assert totals["recall_known_illicit"] == 0
    assert steps.loc[41, "reviewed"] == 0 and pd.isna(
        steps.loc[41, "recall_known_illicit"]
    )
    shuffled = EllipticDataset(
        periods.transactions.iloc[::-1], periods.edges.iloc[::-1]
    )
    other = build_window(shuffled, *WINDOWS["test"])
    _, ranked = review_queue(window, scores, budget=2, seed=42)
    _, reranked = review_queue(other, scores.iloc[::-1], budget=2, seed=42)
    pd.testing.assert_frame_equal(ranked, reranked)
    for step in (40, 49):
        ids = [
            i for i in window.steps[window.steps.eq(step)].index if scores.loc[i] == 0.5
        ]
        first_tie = min(ids, key=lambda i: hashlib.sha256(f"42|{i}".encode()).digest())
        assert ranked.loc[step, "confirmed_illicit"] == int(
            window.labels.loc[first_tie] == "1"
        )


def test_installed_training_cli_persists_the_selected_model_and_aggregate_run(
    model_input: Path, tmp_path: Path
) -> None:
    command = shutil.which("elliptic-train", path=sysconfig.get_path("scripts"))
    assert command is not None
    output = tmp_path / "local runs"
    result = subprocess.run(
        [
            command,
            "--data-dir",
            str(model_input),
            "--output-dir",
            str(output),
            "--budget-per-step",
            "2",
            "--dataset-version",
            "fixture-v1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (run,) = output.iterdir()
    assert str(run) in result.stdout
    assert {p.name for p in run.iterdir()} == {
        "model.joblib",
        "manifest.json",
        "metrics.json",
        "test_by_step.csv",
        ".gitignore",
    }
    metrics = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["dataset"]["kaggle_version"] == "fixture-v1"
    assert (
        manifest["dataset"]["files_sha256"][FEATURES_FILE]
        == hashlib.sha256((model_input / FEATURES_FILE).read_bytes()).hexdigest()
    )
    assert (
        manifest["model_sha256"]
        == hashlib.sha256((run / "model.joblib").read_bytes()).hexdigest()
    )
    assert list(manifest["features_in_order"]) == list(FEATURE_AUDIT)
    assert set(metrics["validation"]["candidate_average_precision"]) == {
        "logistic_regression",
        "xgboost",
    }
    assert manifest["threshold"] == metrics["threshold"]
    assert manifest["winner"] == metrics["selected_model"]
    test = build_window(load_dataset(model_input), *WINDOWS["test"])
    scores = predict_scores(joblib.load(run / "model.joblib"), test.features)
    assert metrics["test"]["operating_point"] == at_threshold(
        test.known_target(), scores, manifest["threshold"]
    )
    assert metrics["test"]["average_precision"] == average_precision(
        test.known_target(), scores
    )
    assert metrics["windows"]["test"]["unknown"] == 2
    assert metrics["test"]["review_queue"]["reviewed"] == 4
    queue = pd.read_csv(run / "test_by_step.csv")
    assert queue["time_step"].tolist() == list(range(40, 50))
    assert queue["reviewed"].sum() == 4
    for filename in ("manifest.json", "metrics.json", "test_by_step.csv"):
        assert test.labels.index[0] not in (run / filename).read_text()


def test_unusable_training_data_preserves_previous_artifacts(
    model_input: Path, tmp_path: Path
) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    previous = output / "previous"
    previous.write_text("keep")
    assert main(["--data-dir", str(model_input), "--output-dir", str(model_input)]) == 1
    assert (
        main(
            [
                "--data-dir",
                str(model_input),
                "--output-dir",
                str(output),
                "--budget-per-step",
                "0",
            ]
        )
        == 1
    )
    classes = model_input / CLASSES_FILE
    classes.write_text(classes.read_text().replace(",1\n", ",unknown\n"))
    assert main(["--data-dir", str(model_input), "--output-dir", str(output)]) == 1
    assert list(output.iterdir()) == [previous] and previous.read_text() == "keep"
