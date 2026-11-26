"""Score Elliptic time steps with a trusted local Phase 2 model."""

import argparse
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import xgboost

from .explanations import explain_scores, plot_importance
from .inference import load_frozen_model, predict_scores
from .ingestion import load_scoring_graph
from .model_evaluation import rank_transactions
from .model_features import FEATURE_AUDIT, scoring_features
from .runs import (
    file_digest,
    file_state,
    implementation_digest,
    new_run,
    read_json,
    write_json,
)
from .schema import EDGES_FILE, FEATURES_FILE, TIME_STEPS, DatasetError


def score_batch(
    model_run: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    first_step: int | None = None,
    last_step: int | None = None,
) -> Path:
    """Score all eligible nodes, rank by step, and explain their positive-class scores."""
    if any(
        output_dir.resolve().is_relative_to(p.resolve()) for p in (data_dir, model_run)
    ):
        raise DatasetError(
            "Choose a scoring output directory outside the inputs and model run."
        )
    manifest_path = model_run / "manifest.json"
    paths = [data_dir / FEATURES_FILE, data_dir / EDGES_FILE]
    before = file_state([manifest_path, *paths])
    manifest = read_json(manifest_path)
    try:
        start, end = manifest["windows_inclusive"]["test"]
        budget, seed = (
            manifest["review_budget_per_step"],
            manifest["seed"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise DatasetError(
            "Training manifest lacks a valid window, seed or budget."
        ) from exc
    first = start if first_step is None else first_step
    last = end if last_step is None else last_step
    if (
        type(first) is not int
        or type(last) is not int
        or first not in TIME_STEPS
        or last not in TIME_STEPS
        or first > last
    ):
        raise DatasetError("Choose an inclusive scoring window from 1 to 49.")
    if type(budget) is not int or budget < 1 or type(seed) is not int:
        raise DatasetError("Invalid review budget or seed in the training manifest.")
    frozen = load_frozen_model(model_run, manifest)
    model, threshold = frozen.pipeline, frozen.threshold
    graph = load_scoring_graph(data_dir)
    features, steps = scoring_features(graph, first, last)
    input_hashes = {path.name: file_digest(path) for path in paths}
    training_manifest_hash = file_digest(manifest_path)
    if before != file_state([manifest_path, *paths]):
        raise DatasetError("An input changed during batch ingestion.")
    scores = predict_scores(model, features).astype("float64")
    ranked = rank_transactions(steps, scores, seed)
    ranked["above_threshold"] = ranked["risk_score"].ge(threshold)
    ranked["in_review_queue"] = ranked["review_rank"].le(budget)

    contributions, base, residual = explain_scores(model, features, scores)
    importance = contributions.abs().mean().sort_values(ascending=False)
    queue = (
        ranked.loc[ranked["in_review_queue"]]
        .join(features)
        .join(base)
        .join(contributions.add_prefix("shap_"))
    )
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "model": frozen.name,
        "model_sha256": frozen.sha256,
        "training_manifest_sha256": training_manifest_hash,
        "implementation_sha256": implementation_digest(),
        "input_files_sha256": input_hashes,
        "first_step": first,
        "last_step": last,
        "threshold": threshold,
        "budget_per_step": budget,
        "seed": seed,
        "features": list(FEATURE_AUDIT),
        "scored": len(ranked),
        "above_threshold": int(ranked["above_threshold"].sum()),
        "queued": len(queue),
        "environment": {
            "python": platform.python_version(),
            "packages": {
                **{
                    name: version(name)
                    for name in (
                        "elliptic-fraud",
                        "shap",
                        "scikit-learn",
                        "joblib",
                        "numpy",
                        "pandas",
                    )
                },
                "xgboost": xgboost.__version__,
            },
        },
        "explanation_units": "log-odds",
        "mean_abs_shap": importance.to_dict(),
        "max_score_reconstruction_error": residual,
    }
    with new_run(output_dir, "score") as (staging, destination):
        ranked.to_csv(staging / "scores.csv")
        queue.to_csv(staging / "review_queue.csv")
        write_json(staging / "summary.json", summary)
        plot_importance(importance, staging / "shap_importance.png")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Produce a private batch run using the saved model and validation threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-run", required=True, type=Path, help="trusted local training run"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scores"))
    parser.add_argument(
        "--first-step",
        type=int,
        help="inclusive start, defaults to the model's test start",
    )
    parser.add_argument(
        "--last-step", type=int, help="inclusive end, defaults to the model's test end"
    )
    args = parser.parse_args(argv)
    try:
        run = score_batch(
            args.model_run,
            args.data_dir,
            args.output_dir,
            first_step=args.first_step,
            last_step=args.last_step,
        )
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Scoring run written to {run}")
    return 0
