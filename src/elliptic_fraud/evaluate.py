"""Re-measure a frozen legacy model and write transaction-free evaluation reports."""

import argparse
import platform
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from .folds import DEV_STEPS
from .inference import load_frozen_model, predict_scores
from .ingestion import load_dataset
from .model_evaluation import (
    at_threshold,
    block_bootstrap_ci,
    brier_score,
    per_step_ap,
    pooled_ap,
    precision_recall_at_k_labelled,
    queue_vs_random,
    review_queue,
    roc_auc,
    tie_mass_at_cut,
)
from .model_features import FEATURE_AUDIT, WINDOWS, Window, build_window
from .runs import (
    file_digest,
    file_state,
    implementation_digest,
    new_run,
    read_json,
    snapshot_digest,
    write_json,
)
from .schema import EXPECTED_FILES, DatasetError

LABELLED_CAPACITIES = (10, 25, 50)
QUEUE_CAPACITIES = (25, 50, 100)
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42


def evaluation_environment() -> dict:
    """Record the package versions that influence numerical results."""
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl", "joblib")
    return {
        "python": platform.python_version(),
        "packages": {
            **{name: version(name) for name in packages},
            "xgboost": xgboost.__version__,
        },
    }


def _queue_metrics(window: Window, scores: pd.Series, budget: int, seed: int) -> dict:
    totals, per_step = review_queue(window, scores, budget, seed)
    ties = tie_mass_at_cut(window.steps, scores, window.first, window.last, budget)
    rows = {}
    for step, values in (
        per_step.drop(columns="recall_known_illicit").to_dict("index").items()
    ):
        baseline = queue_vs_random(per_step.loc[[step]])
        reviewed = values["reviewed"]
        positives = values["known_illicit"]
        rows[step] = {
            **values,
            "unknown_share": values["reviewed_unknown"] / reviewed
            if reviewed
            else None,
            "recall_known_illicit": values["confirmed_illicit"] / positives
            if positives
            else None,
            **baseline,
            **ties[step],
        }
    return {
        **totals,
        "unknown_share": totals["reviewed_unknown"] / totals["reviewed"]
        if totals["reviewed"]
        else None,
        **queue_vs_random(per_step),
        "steps_with_ties_spanning_cut": sum(
            step["spans_cut"] for step in ties.values()
        ),
        "by_step": rows,
    }


def evaluate_window(
    window: Window,
    scores: pd.Series,
    *,
    seed: int,
    threshold: float,
    budget: int = 50,
) -> dict:
    """Aggregate fixed-rule metrics. Unknown outcomes enter only the queue and support."""
    target = window.known_target()
    step_ap = per_step_ap(window, scores)
    defined = [ap for ap in step_ap.values() if ap is not None]
    ap_ci = block_bootstrap_ci(step_ap, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    labelled_rankings = {}
    for k in LABELLED_CAPACITIES:
        rows = precision_recall_at_k_labelled(window, scores, k, seed)
        precision = {step: row["precision"] for step, row in rows.items()}
        recall = [row["recall"] for row in rows.values() if row["recall"] is not None]
        precision_ci = block_bootstrap_ci(
            precision, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
        )
        labelled_rankings[k] = {
            "mean_step_precision": precision_ci["estimate"],
            "mean_step_recall": statistics.mean(recall) if recall else None,
            "precision_ci": precision_ci,
            "by_step": rows,
        }
    report = {
        "support": window.support(),
        "pooled_ap": pooled_ap(target, scores),
        "per_step_ap": step_ap,
        "mean_step_ap": ap_ci["estimate"],
        "step_ap_std": statistics.stdev(defined) if len(defined) > 1 else None,
        "mean_step_ap_ci": ap_ci,
        "roc_auc": roc_auc(target, scores),
        "brier_score": brier_score(target, scores),
        "distinct_scores": int(scores.nunique()),
        "threshold": threshold,
        "operating_point": at_threshold(target, scores, threshold),
        "q_lab": labelled_rankings,
        "q_all": {
            k: _queue_metrics(window, scores, k, seed)
            for k in sorted(set(QUEUE_CAPACITIES) | {budget})
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
    }
    if window.first < 43 <= window.last:
        before = window.steps.loc[target.index].lt(43)
        report["split_at_43"] = {
            "before_43_pooled_ap": pooled_ap(target.loc[before], scores),
            "from_43_pooled_ap": pooled_ap(target.loc[~before], scores),
        }
    return report


def _validate_score_run(
    score_run: Path,
    model_sha: str,
    manifest_sha: str,
    window: Window,
    scores: pd.Series,
) -> None:
    summary = read_json(score_run / "summary.json")
    if (
        summary.get("model_sha256") != model_sha
        or summary.get("training_manifest_sha256") != manifest_sha
        or (summary.get("first_step"), summary.get("last_step"))
        != (window.first, window.last)
    ):
        raise DatasetError(
            "The score run must be the replay of this model and test window."
        )
    try:
        reference = pd.read_csv(
            score_run / "scores.csv",
            usecols=["tx_id", "time_step", "risk_score"],
            dtype={"tx_id": scores.index.dtype},
            float_precision="round_trip",
        ).set_index("tx_id")
    except ValueError:
        raise DatasetError("The score run contains an invalid score table.") from None
    if not reference.index.is_unique or not reference.index.sort_values().equals(
        scores.index.sort_values()
    ):
        raise DatasetError("The score run does not cover the frozen test window.")
    aligned = reference.loc[scores.index]
    if not np.array_equal(
        aligned["time_step"], window.steps.loc[scores.index]
    ) or not np.array_equal(aligned["risk_score"], scores):
        raise DatasetError("The scores differ from the saved batch replay.")


def _assert_legacy_parity(recorded: dict, evaluated: dict, budget: int) -> None:
    observed = evaluated["q_all"][budget]
    if (
        recorded["average_precision"] != evaluated["pooled_ap"]
        or recorded["operating_point"] != evaluated["operating_point"]
        or any(
            observed.get(key) != value
            for key, value in recorded["review_queue"].items()
        )
    ):
        raise DatasetError(
            "Re-measurement did not reproduce the legacy AP, operating point or queue."
        )


def evaluate_legacy(
    model_run: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    score_run: Path | None = None,
    prepare_development: bool = False,
) -> Path:
    """Score one approved legacy replay and write a new aggregate-only report."""
    protected = [
        data_dir,
        model_run,
        *(Path("artifacts") / name for name in ("models", "scores", "eda", "tracking")),
    ]
    if score_run is not None:
        protected.append(score_run)
    if any(
        output_dir.resolve().is_relative_to(directory.resolve())
        for directory in protected
    ):
        raise DatasetError(
            "Choose a new evaluations location outside protected input and artifact directories."
        )
    manifest_path = model_run / "manifest.json"
    manifest = read_json(manifest_path)
    recorded = read_json(model_run / "metrics.json")
    try:
        windows = {
            key: tuple(value) for key, value in manifest["windows_inclusive"].items()
        }
        seed, budget = manifest["seed"], manifest["review_budget_per_step"]
        snapshot = manifest["dataset"]
    except (KeyError, TypeError, AttributeError):
        raise DatasetError(
            "The legacy manifest is missing its evaluation provenance."
        ) from None
    if (
        windows != WINDOWS
        or type(seed) is not int
        or type(budget) is not int
        or budget < 1
    ):
        raise DatasetError(
            "Legacy evaluation requires the original temporal windows, seed and capacity."
        )
    model = load_frozen_model(model_run, manifest)
    paths = [data_dir / name for name in EXPECTED_FILES]
    before = file_state(paths)
    fingerprint = {path.name: file_digest(path) for path in paths}
    snapshot_sha = snapshot_digest(fingerprint)
    if (
        not isinstance(snapshot, dict)
        or fingerprint != snapshot.get("files_sha256")
        or snapshot_sha != snapshot.get("snapshot_sha256")
    ):
        raise DatasetError(
            "Evaluation inputs must match the legacy model's dataset snapshot."
        )
    dataset = load_dataset(data_dir)
    if before != file_state(paths):
        raise DatasetError("The input snapshot changed during evaluation.")
    window = build_window(dataset, *WINDOWS["test"])
    scores = predict_scores(model.pipeline, window.features).astype("float64")
    manifest_sha = file_digest(manifest_path)
    if score_run is not None:
        _validate_score_run(score_run, model.sha256, manifest_sha, window, scores)
    evaluated = evaluate_window(
        window, scores, seed=seed, threshold=model.threshold, budget=budget
    )
    _assert_legacy_parity(recorded["test"], evaluated, budget)
    if prepare_development:
        from .experiments import cache_development

        development = build_window(dataset, *DEV_STEPS)
        cache_development(development, snapshot)
    with new_run(output_dir, "evaluation") as (staging, destination):
        write_json(
            staging / "evaluation.json",
            {
                "schema_version": 2,
                "purpose": "re-measurement of legacy run, no selection",
                "created_utc": datetime.now(UTC).isoformat(),
                "dataset": snapshot,
                "model": model.name,
                "model_sha256": model.sha256,
                "training_manifest_sha256": manifest_sha,
                "implementation_sha256": implementation_digest(),
                "environment": evaluation_environment(),
                "feature_set": "T0",
                "features_in_order": list(FEATURE_AUDIT),
                "label_policy": manifest["label_policy"],
                "feature_availability": manifest["availability"],
                "seed": seed,
                "legacy_budget_per_step": budget,
                "score_run_verified": score_run is not None,
                "test": evaluated,
            },
        )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Re-measure the frozen model for an explicitly approved test access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--score-run",
        type=Path,
        help="verify exact scores and time steps against the saved batch replay",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluations")
    )
    parser.add_argument(
        "--prepare-development",
        action="store_true",
        help="cache only steps 1–39 during this approved read",
    )
    args = parser.parse_args(argv)
    try:
        output = evaluate_legacy(
            args.model_run,
            args.data_dir,
            args.output_dir,
            score_run=args.score_run,
            prepare_development=args.prepare_development,
        )
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Aggregate evaluation written to {output}")
    return 0
