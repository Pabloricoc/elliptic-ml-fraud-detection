"""Run the phase 3 feature-tier experiment on a verified development export."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .ablation_report import (
    paired_comparison,
    render_ablation,
    select_incumbent,
    summarize_arm,
)
from .analysis import a1_probe
from .development import DEVELOPMENT_DIR, load_development_export
from .evaluate import evaluate_window, evaluation_environment
from .experiments import DEVELOPMENT_THRESHOLD, clean_git_revision, record_fold_mlflow
from .feature_cache import cached_development_features
from .folds import DEV_STEPS, Fold, development_folds, slice_development
from .inference import predict_scores
from .ingestion import EllipticDataset
from .model_features import FeatureSet, Window, get_feature_set
from .runs import implementation_digest, read_json, snapshot_digest, write_json
from .schema import DatasetError
from .train import BOOST_PARAMS, SEED

FAMILY_PARAMS = {
    "xgboost": BOOST_PARAMS,
    "random_forest": {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "n_jobs": 6,
        "random_state": SEED,
    },
}
FAMILIES = {"xgboost": XGBClassifier, "random_forest": RandomForestClassifier}


@dataclass(frozen=True)
class TierComparison:
    feature_set: FeatureSet
    against: str | None


def _plan(document: dict) -> list[TierComparison]:
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != {"S1", "S2"}:
        raise DatasetError("The ablation config must define the S1 and S2 scenarios.")
    plan = []
    for scenario, reference in (("S1", "T0"), ("S2", "T1b")):
        entries = scenarios[scenario]
        seen = set()
        if not isinstance(entries, list) or not entries:
            raise DatasetError(f"The {scenario} ablation requires a baseline.")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"features", "against"}:
                raise DatasetError("Each tier requires features and against fields.")
            name, against = entry["features"], entry["against"]
            if (
                not isinstance(name, str)
                or name in seen
                or (against is not None and not isinstance(against, str))
                or (not seen and (name != reference or against is not None))
                or (seen and against not in seen)
            ):
                raise DatasetError(
                    "Declare the baseline first and compare distinct tiers to earlier references."
                )
            if scenario == "S1" and name not in {"T0", "T1"}:
                raise DatasetError("S1 is restricted to T0 and T1.")
            plan.append(TierComparison(get_feature_set(name, scenario), against))
            seen.add(name)
    return plan


def _fold_report(
    output_dir: Path,
    development: Window,
    fold: Fold,
    family: str,
    feature_set: FeatureSet,
    provenance: dict,
    removed: tuple[str, ...],
) -> dict:
    columns = [column for column in feature_set.columns if column not in removed]
    if not columns:
        raise DatasetError("The A1 mask would remove every predictor.")
    config = {
        **{name: value for name, value in provenance.items() if name != "git_commit"},
        "schema_version": 2,
        "scenario": feature_set.scenario,
        "feature_set": feature_set.name,
        "feature_version": feature_set.version,
        "columns_sha256": snapshot_digest({"columns": columns}),
        "family": family,
        "params": FAMILY_PARAMS[family],
        "fold": asdict(fold),
        "embargo": 0,
        "seed": SEED,
        "prediction_jobs": 1,
        "diagnostic_threshold": DEVELOPMENT_THRESHOLD,
    }
    digest = snapshot_digest(config)
    folder = output_dir / digest
    path = folder / "evaluation.json"
    with FileLock(output_dir / f".{digest}.lock"):
        if not path.exists():
            train = slice_development(development, fold.train)
            validation = slice_development(development, fold.validation)
            target = train.known_target()
            if target.nunique() != 2:
                raise DatasetError("Each training fold needs both known classes.")
            model = Pipeline(
                [("classifier", FAMILIES[family](**FAMILY_PARAMS[family]))]
            )
            model.fit(train.features.loc[target.index, columns], target)
            # Fix the order of tree probability accumulation during forest scoring.
            if family == "random_forest":
                model.set_params(classifier__n_jobs=1)
            scores = predict_scores(model, validation.features.loc[:, columns]).astype(
                "float64"
            )
            metrics = evaluate_window(
                validation, scores, seed=SEED, threshold=DEVELOPMENT_THRESHOLD
            )
            report = {
                "config_sha256": digest,
                "config": config,
                "git_commit": provenance["git_commit"],
                "columns": columns,
                "label_policy": "1 illicit, 2 licit, unknown unlabelled",
                "feature_availability": feature_set.availability,
                "availability_assumption": "conditional on A1"
                if feature_set.scenario == "S2"
                else None,
                "train": train.support(),
                "validation": metrics,
            }
            folder.mkdir(exist_ok=True)
            temporary = folder / ".evaluation.partial"
            write_json(temporary, report)
            temporary.replace(path)
        report = read_json(path)
        if (
            report.get("config_sha256") != digest
            or snapshot_digest(report.get("config", {})) != digest
        ):
            raise DatasetError(
                "A fold report does not match its experiment configuration."
            )
    report["mlflow_run_id"] = record_fold_mlflow(output_dir, report)
    return report


def _feature_window(
    dataset: EllipticDataset, feature_set: FeatureSet, snapshot: str, cache_root: Path
) -> Window:
    features, steps = cached_development_features(
        dataset, snapshot, feature_set, cache_root=cache_root
    )
    labels = dataset.transactions.set_index("tx_id")["class"].loc[features.index]
    return Window(features, labels, steps, *DEV_STEPS)


def _removed_columns(feature_set: FeatureSet, flagged: list[str]) -> tuple[str, ...]:
    return tuple(
        column
        for column in feature_set.columns
        if column in flagged
        or any(column.endswith(f"_{feature}") for feature in flagged)
    )


def run_ablation(
    config_path: Path, development_dir: Path, output_dir: Path, *, git_commit: str
) -> Path:
    """Fit the configured arms and publish aggregate results and a Markdown report."""
    document = read_json(config_path)
    plan = _plan(document)
    output_dir = output_dir.resolve()
    cache_root = (development_dir.resolve().parent / "phase3-features").resolve()
    protected = [
        development_dir.resolve(),
        Path("data/raw").resolve(),
        *(
            (Path("artifacts") / name).resolve()
            for name in ("models", "scores", "eda", "tracking")
        ),
    ]
    if any(
        path.is_relative_to(root)
        for path in (output_dir, cache_root)
        for root in protected
    ):
        raise DatasetError(
            "Choose ablation output paths outside the export, raw data and protected artifacts."
        )
    dataset, manifest = load_development_export(development_dir)
    snapshot = manifest["development_snapshot_sha256"]
    provenance = {
        "development_snapshot_sha256": snapshot,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "implementation_sha256": implementation_digest(),
        "environment": evaluation_environment(),
        "git_commit": git_commit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = development_folds()
    arms = {}
    for entry in plan:
        features = entry.feature_set
        window = _feature_window(dataset, features, snapshot, cache_root)
        for family in FAMILIES:
            reports = [
                _fold_report(output_dir, window, fold, family, features, provenance, ())
                for fold in folds
            ]
            arm = summarize_arm(features.scenario, features.name, family, reports)
            arm["paired_tier"] = (
                paired_comparison(
                    arm, arms[features.scenario, entry.against, family], original=True
                )
                if entry.against is not None
                else None
            )
            arms[features.scenario, features.name, family] = arm
            print(f"Completed {features.scenario} {features.name} {family}", flush=True)

    selection = {}
    for scenario in ("S1", "S2"):
        incumbent, comparisons = select_incumbent(
            [arm for arm in arms.values() if arm["scenario"] == scenario], original=True
        )
        selection[scenario] = {
            "incumbent": incumbent,
            "ties_ok": incumbent["validation_steps"] > 0
            and 10 * incumbent["small_ties"] >= 9 * incumbent["validation_steps"],
            "comparisons": comparisons,
        }

    probe = {fold.name: a1_probe(dataset, fold.train) for fold in folds}
    applicable = {}
    for key, arm in arms.items():
        if arm["scenario"] == "S2" and arm["mean_step_ap"] is not None:
            features = get_feature_set(arm["features"], "S2")
            masks = {
                fold.name: _removed_columns(features, probe[fold.name]["flagged"])
                for fold in folds
            }
            if any(masks.values()):
                applicable[key] = masks
    sensitivity = None
    if applicable:
        winner = selection["S2"]["incumbent"]
        winner_key = ("S2", winner["features"], winner["family"])
        source_key = (
            winner_key
            if winner_key in applicable
            else min(
                applicable,
                key=lambda key: (-arms[key]["mean_step_ap"], arms[key]["complexity"]),
            )
        )
        scenario, feature_name, family = source_key
        features = get_feature_set(feature_name, scenario)
        masks = applicable[source_key]
        window = _feature_window(dataset, features, snapshot, cache_root)
        reports = [
            _fold_report(
                output_dir, window, fold, family, features, provenance, masks[fold.name]
            )
            for fold in folds
        ]
        filtered = summarize_arm(
            "S2", f"{feature_name} minus A1 flags", family, reports
        )
        source = arms[source_key]
        filtered["paired_tier"] = paired_comparison(filtered, source, original=True)
        sensitivity = {
            "source": source,
            "filtered": filtered,
            "removed": {name: len(mask) for name, mask in masks.items()},
        }

    identity = snapshot_digest({"plan": document, **provenance})
    report_json = output_dir / f"phase3-{identity}.json"
    summary = {
        **provenance,
        "config": document,
        "arms": list(arms.values()),
        "selection": selection,
        "probe": probe,
        "sensitivity": sensitivity,
        "artifact": f"artifacts/experiments/{report_json.name}",
    }
    with FileLock(output_dir / f".{identity}.lock"):
        temporary = report_json.with_suffix(".partial")
        write_json(temporary, summary)
        temporary.replace(report_json)
        markdown = report_json.with_suffix(".md")
        partial_markdown = markdown.with_suffix(".md.partial")
        partial_markdown.write_text(render_ablation(summary), encoding="utf-8")
        partial_markdown.replace(markdown)
    return report_json


def load_cached_ablation(report_path: Path) -> dict:
    """Verify the aggregate and saved fold reports without loading a dataset."""
    key = report_path.stem.removeprefix("phase3-")
    if (
        report_path.suffix != ".json"
        or not report_path.stem.startswith("phase3-")
        or len(key) != 64
        or any(char not in "0123456789abcdef" for char in key)
    ):
        raise DatasetError("Choose a saved phase 3 aggregate JSON.")
    summary = read_json(report_path)
    folds = development_folds()
    provenance = (
        "development_snapshot_sha256",
        "source_snapshot_sha256",
        "implementation_sha256",
        "environment",
    )
    expected_key = snapshot_digest(
        {
            "plan": summary["config"],
            **{field: summary[field] for field in (*provenance, "git_commit")},
        }
    )
    if key != expected_key:
        raise DatasetError("The aggregate does not match its configuration identifier.")
    declared = _plan(summary["config"])
    indexed = {
        (arm["scenario"], arm["features"], arm["family"]): arm
        for arm in summary["arms"]
    }
    if list(indexed) != [
        (entry.feature_set.scenario, entry.feature_set.name, family)
        for entry in declared
        for family in FAMILIES
    ]:
        raise DatasetError("The saved arms do not match the declared ablation plan.")
    for entry in declared:
        scenario, name = entry.feature_set.scenario, entry.feature_set.name
        for family in FAMILIES:
            arm = indexed[scenario, name, family]
            original_pair = (
                paired_comparison(
                    arm, indexed[scenario, entry.against, family], original=True
                )
                if entry.against is not None
                else None
            )
            if arm["paired_tier"] != original_pair:
                raise DatasetError(
                    "A saved tier comparison differs from its declared reference."
                )
    arms = [(arm, arm["features"]) for arm in summary["arms"]]
    sensitivity = summary["sensitivity"]
    if sensitivity:
        arms.append((sensitivity["filtered"], sensitivity["source"]["features"]))
    for arm, name in arms:
        reports = []
        for reference, fold in zip(arm["folds"], folds, strict=True):
            digest = reference["config_sha256"]
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise DatasetError("Invalid cached fold identifier.")
            report = read_json(report_path.parent / digest / "evaluation.json")
            config = report["config"]
            if (
                report["config_sha256"] != digest
                or snapshot_digest(config) != digest
                or any(config[field] != summary[field] for field in provenance)
                or config["scenario"] != arm["scenario"]
                or config["feature_set"] != name
                or config["family"] != arm["family"]
                or config["fold"]
                != {
                    "name": fold.name,
                    "train": list(fold.train),
                    "validation": list(fold.validation),
                }
                or report["validation"]["support"]["last_step"] != fold.validation[1]
            ):
                raise DatasetError("A cached fold does not match the saved ablation.")
            reports.append({**report, "mlflow_run_id": reference["mlflow_run_id"]})
        computed = summarize_arm(
            arm["scenario"], arm["features"], arm["family"], reports
        )
        if any(computed[field] != arm[field] for field in computed):
            raise DatasetError("Saved arm metrics differ from the cached fold reports.")
    return summary


def review_cached_ablation(report_path: Path) -> Path:
    """Regenerate decisions from a verified aggregate and its saved folds."""
    summary = load_cached_ablation(report_path)
    review_path = report_path.with_suffix(".review.md")
    with FileLock(review_path.with_suffix(".lock")):
        temporary = review_path.with_suffix(".partial")
        temporary.write_text(render_ablation(summary), encoding="utf-8")
        temporary.replace(review_path)
    return review_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the phase 3 fold plan from JSON and a frozen development export."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument(
        "--review-results",
        type=Path,
        help="regenerate the report from cached phase 3 folds",
    )
    parser.add_argument("--development-dir", type=Path, default=DEVELOPMENT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args(argv)
    try:
        if args.review_results is not None:
            review = review_cached_ablation(args.review_results)
            print(f"Reviewed ablation written to {review}.")
            return 0
        report = run_ablation(
            args.config,
            args.development_dir,
            args.output_dir,
            git_commit=clean_git_revision(),
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        f"Aggregate ablation and report written to {report} and {report.with_suffix('.md')}."
    )
    return 0
