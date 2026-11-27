import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from elliptic_fraud import ablation, development
from elliptic_fraud.ablation_report import paired_comparison, select_incumbent
from elliptic_fraud.analysis import a1_probe
from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.runs import read_json
from elliptic_fraud.schema import FEATURE_COLUMNS, FEATURES_FILE, DatasetError


def test_a1_probe_controls_degree_and_ignores_validation_rows() -> None:
    ids = ["1", "2", "3", "4", "5", "6"]
    edges = pd.DataFrame(
        [("1", "2"), ("1", "3"), ("2", "3"), ("3", "4"), ("5", "6")],
        columns=["source_id", "target_id"],
    )
    rows = pd.DataFrame(0.0, index=range(len(ids)), columns=FEATURE_COLUMNS).assign(
        tx_id=ids, time_step=1, **{"class": "unknown"}
    )
    rows["feature_001"] = rows["tx_id"].map(edges["source_id"].value_counts()).fillna(0)
    rows["feature_002"] = rows["tx_id"].map(edges["target_id"].value_counts()).fillna(0)
    report = a1_probe(EllipticDataset(rows, edges), (1, 1))
    assert report["partial_spearman"]["feature_001"] == pytest.approx(1.0)
    assert report["partial_spearman"]["feature_002"] is None
    assert report["flagged"] == ["feature_001"]
    later = rows.assign(
        tx_id=[f"9{tx_id}" for tx_id in ids], time_step=2, feature_001=-100
    )
    combined = EllipticDataset(pd.concat([rows, later], ignore_index=True), edges)
    assert a1_probe(combined, (1, 1)) == report
    with pytest.raises(DatasetError, match="development training window"):
        a1_probe(combined, (1, 40))


def test_selection_applies_paired_improvement_and_both_operational_gates() -> None:
    baseline = {
        "features": "T0",
        "family": "xgboost",
        "fold_ap": [0.2] * 4,
        "p_lab_50": 0.3,
        "validation_steps": 20,
        "small_ties": 20,
        "complexity": [4, 100],
    }
    candidate = {
        **baseline,
        "features": "T1",
        "fold_ap": [0.3, 0.3, 0.31, 0.29],
        "small_ties": 18,
    }
    selected, comparisons = select_incumbent([baseline, candidate])
    assert selected == candidate and comparisons[0]["decision"] == "improves"
    assert (
        paired_comparison({**candidate, "small_ties": 17}, baseline)["decision"]
        == "retained incumbent"
    )
    assert (
        paired_comparison({**candidate, "p_lab_50": 0.29}, baseline)["decision"]
        == "retained incumbent"
    )
    assert (
        paired_comparison({**candidate, "fold_ap": [0.5, 0.5, 0.1, 0.1]}, baseline)[
            "decision"
        ]
        == "retained incumbent"
    )
    assert (
        paired_comparison(
            {**candidate, "fold_ap": [0.201, 0.201, 0.201, 0.1]}, baseline
        )["decision"]
        == "retained incumbent"
    )
    simpler = {**candidate, "fold_ap": [0.2] * 4, "complexity": [3, 100]}
    assert paired_comparison(simpler, baseline)["decision"] == "simpler within 1 SE"


@pytest.mark.parametrize("periods", [tuple(range(1, 40)) + (40,)], indirect=True)
def test_configured_runner_uses_export_dedupes_and_repeats_the_a1_mask(
    model_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_path = model_input / FEATURES_FILE
    features = pd.read_csv(feature_path, header=None, dtype={0: str})
    features[2] = features[0].str[-1].map({"0": 1, "1": 1, "2": 0, "3": 1})
    features.to_csv(feature_path, index=False, header=False)
    export = development.export_development(model_input, tmp_path / "development")

    def unexpected(*args, **kwargs):
        pytest.fail("An unapproved raw read or duplicate fit was attempted.")

    monkeypatch.setattr(development, "load_dataset", unexpected)
    monkeypatch.setattr(ablation, "clean_git_revision", lambda: "synthetic")
    for family, params in ablation.FAMILY_PARAMS.items():
        monkeypatch.setitem(
            ablation.FAMILY_PARAMS, family, {**params, "n_estimators": 3, "n_jobs": 1}
        )
    plan = tmp_path / "ablation.json"
    document = {
        "scenarios": {
            "S1": [{"features": "T0", "against": None}],
            "S2": [
                {"features": "T1b", "against": None},
                {"features": "T1b+T2", "against": "T1b"},
            ],
        }
    }
    plan.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "experiments"
    args = [
        "--config",
        str(plan),
        "--development-dir",
        str(export),
        "--output-dir",
        str(output),
    ]
    assert ablation.main(args) == 0
    (aggregate,) = output.glob("phase3-*.json")
    summary = read_json(aggregate)
    assert len(summary["arms"]) == 6
    assert summary["sensitivity"]["removed"] == dict.fromkeys(
        ("F1", "F2", "F3", "F4"), 1
    )
    assert "conditional on A1" in aggregate.with_suffix(".md").read_text(
        encoding="utf-8"
    )
    for arm in summary["arms"]:
        assert len(arm["folds"]) == 4
        for run in arm["folds"]:
            report = read_json(output / run["config_sha256"] / "evaluation.json")
            assert (
                report["train"]["last_step"]
                < report["validation"]["support"]["first_step"]
            )
            assert report["validation"]["support"]["last_step"] <= 39
            assert report["train"]["transactions"] > report["train"]["labelled"]
    monkeypatch.setattr(Pipeline, "fit", unexpected)
    assert ablation.main(args) == 0
    assert read_json(aggregate) == summary

    document["scenarios"]["S1"].append({"features": "T2", "against": "T0"})
    plan.write_text(json.dumps(document), encoding="utf-8")
    assert ablation.main(args) == 1
