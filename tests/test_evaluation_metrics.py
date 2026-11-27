import pandas as pd
import pytest

from elliptic_fraud.evaluate import evaluate_window
from elliptic_fraud.model_evaluation import (
    block_bootstrap_ci,
    precision_recall_at_k_labelled,
    roc_auc,
    tie_mass_at_cut,
)
from elliptic_fraud.model_features import Window


def test_evaluation_distinguishes_labelled_ranking_from_operational_queue() -> None:
    index = pd.Index(["100", "101", "102", "103", "201", "202", "301"], name="tx_id")
    steps = pd.Series([1, 1, 1, 1, 2, 2, 3], index=index)
    labels = pd.Series(
        ["1", "2", "unknown", "2", "2", "unknown", "unknown"], index=index
    )
    scores = pd.Series([0.8, 0.8, 0.95, 0.2, 0.3, 0.97, 0.99], index=index)
    window = Window(pd.DataFrame(index=index), labels, steps, 1, 4)
    report = evaluate_window(window, scores, seed=42, threshold=0.5, budget=2)

    assert report["pooled_ap"] == 0.5
    assert report["per_step_ap"] == {1: 0.5, 2: None, 3: None, 4: None}
    assert report["mean_step_ap"] == 0.5
    assert report["step_ap_std"] is None
    assert report["roc_auc"] == pytest.approx(5 / 6)
    assert report["brier_score"] == pytest.approx(0.2025)
    assert roc_auc(window.known_target().loc[["100"]], scores) is None

    labelled = precision_recall_at_k_labelled(window, scores, k=2, seed=42)
    assert labelled[1] == {
        "reviewed_labelled": 2,
        "confirmed_illicit": 1,
        "precision": 0.5,
        "recall": 1.0,
    }
    assert labelled[2]["precision"] == 0.0 and labelled[2]["recall"] is None
    assert (
        labelled[3]
        == labelled[4]
        == {
            "reviewed_labelled": 0,
            "confirmed_illicit": 0,
            "precision": None,
            "recall": None,
        }
    )
    assert report["q_lab"][10]["by_step"][1]["reviewed_labelled"] == 3

    queue = report["q_all"][2]
    assert queue["reviewed"] == 5 and queue["reviewed_unknown"] == 3
    assert queue["steps_with_ties_spanning_cut"] == 1
    assert queue["expected_random_hits"] == 0.5
    assert queue["hits_over_expected"] == queue["confirmed_illicit"] / 0.5
    assert queue["by_step"][2]["hits_over_expected"] is None
    assert queue["by_step"][4]["unknown_share"] is None
    assert tie_mass_at_cut(steps, scores, 1, 4, 2)[1] == {
        "strictly_above": 1,
        "tied_at_cut": 2,
        "tied_selected": 1,
        "spans_cut": True,
    }
    assert not tie_mass_at_cut(steps, scores, 1, 4, 10)[1]["spans_cut"]


def test_bootstrap_resamples_steps_and_handles_undefined_draws() -> None:
    distinct_steps = {1: 0.25, 2: 0.75}
    assert block_bootstrap_ci(distinct_steps) == {
        "estimate": 0.5,
        "lower": 0.25,
        "upper": 0.75,
        "defined_draws": 1000,
    }
    sparse = block_bootstrap_ci({1: None, 2: 0.75, 3: None})
    assert sparse["estimate"] == sparse["lower"] == sparse["upper"] == 0.75
    assert 0 < sparse["defined_draws"] < 1000
    assert sparse == block_bootstrap_ci({3: None, 2: 0.75, 1: None})
    assert block_bootstrap_ci({1: None}) == {
        "estimate": None,
        "lower": None,
        "upper": None,
        "defined_draws": 0,
    }
