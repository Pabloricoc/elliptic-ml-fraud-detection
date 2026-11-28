"""Prepare, verify and evaluate the registered Phase 5 shortlist once."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from .development import DEVELOPMENT_DIR, load_development_export
from .evaluate import evaluation_environment
from .experiments import clean_git_revision, record_fold_mlflow
from .ingestion import EllipticDataset, load_dataset
from .model_features import Window, build_feature_set
from .phase5_models import (
    ARM_NAMES,
    LEGACY_RUN,
    S1_TIER,
    S2_TIER,
    TEST_STEPS,
    Arm,
    freeze_models,
    legacy_model,
    load_arms,
)
from .phase5_report import (
    final_metrics,
    paired_contrasts,
    render_evaluation,
    step_draws,
)
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
from .scoring import ranked_batch_scores
from .train import SEED

ROOT = Path("artifacts/experiments/phase5-final")
RAW = Path("data/raw")
PROTOCOL = Path("docs/experiments/preregistration.md")
APPROVAL = "52d2915f3a5b85986a7618ee89d572a53630ac8f"
DEV_SHA = "680aa81f91bfb708de8d8ea01f5eb260e723a0cd59cb7b71cd304526b90c6ca8"
SOURCE_SHA = "903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043"
ENVIRONMENT = {
    "python": "3.12.14",
    "packages": {
        "numpy": "2.3.5",
        "pandas": "2.3.3",
        "scikit-learn": "1.7.2",
        "scipy": "1.16.3",
        "threadpoolctl": "3.6.0",
        "joblib": "1.5.3",
        "xgboost": "3.1.2",
        "lightgbm": "4.7.0",
    },
}


def _provenance() -> dict:
    revision = clean_git_revision()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", APPROVAL, revision], check=True
    )
    diff = subprocess.check_output(
        ["git", "diff", APPROVAL, revision, "--", str(PROTOCOL)]
    )
    if diff:
        raise DatasetError("The approved pre-registration differs from this revision.")
    environment = evaluation_environment()
    environment["packages"]["lightgbm"] = version("lightgbm")
    if environment != ENVIRONMENT:
        raise DatasetError("Use the registered numerical environment.")
    protected = [
        RAW,
        *(Path("artifacts") / name for name in ("models", "scores", "eda", "tracking")),
    ]
    if any(ROOT.resolve().is_relative_to(path.resolve()) for path in protected):
        raise DatasetError("The Phase 5 output resolves into a protected directory.")
    return {
        "git_commit": revision,
        "approval_commit": APPROVAL,
        "approval_date": "2026-11-28",
        "evaluation_command": "python -m elliptic_fraud.phase5 evaluate --frozen PATH --freeze-sha SHA256",
        "preregistration_sha256": file_digest(PROTOCOL),
        "implementation_sha256": implementation_digest(),
        "development_snapshot_sha256": DEV_SHA,
        "source_snapshot_sha256": SOURCE_SHA,
        "legacy_manifest_sha256": file_digest(LEGACY_RUN / "manifest.json"),
        "environment": environment,
    }


def _prepare() -> Path:
    provenance = _provenance()
    legacy = legacy_model(LEGACY_RUN)
    dataset, manifest = load_development_export(DEVELOPMENT_DIR)
    for key in ("development_snapshot_sha256", "source_snapshot_sha256"):
        if manifest[key] != provenance[key]:
            raise DatasetError(
                "The development export differs from the registered release."
            )
    return freeze_models(dataset, ROOT, provenance, legacy)


def _verify_frozen(path: Path, expected_sha: str) -> tuple[dict, dict[str, Arm]]:
    path = path.resolve()
    if path.parent != ROOT.resolve() or not path.name.startswith("frozen-"):
        raise DatasetError(
            "Choose a frozen run inside the Phase 5 experiment directory."
        )
    manifest = path / "freeze.json"
    digest = file_digest(manifest)
    if digest != expected_sha:
        raise DatasetError("The frozen shortlist has a different digest.")
    freeze = read_json(manifest)
    for key, value in _provenance().items():
        if freeze.get(key) != value:
            raise DatasetError(f"The frozen shortlist has different {key}.")
    legacy = legacy_model(LEGACY_RUN)
    return freeze, load_arms(path, freeze, legacy)


def _smoke_development(arms: dict[str, Arm]) -> None:
    dataset, manifest = load_development_export(DEVELOPMENT_DIR)
    if manifest["development_snapshot_sha256"] != DEV_SHA:
        raise DatasetError(
            "The development smoke check requires the registered export."
        )
    frames = {
        tier.scenario: build_feature_set(dataset, tier, 39, 39)[0]
        for tier in (S1_TIER, S2_TIER)
    }
    for arm in arms.values():
        frame = frames[arm.feature_set.scenario]
        scores = arm.score(frame)
        if not scores.index.equals(frame.index) or not scores.between(0, 1).all():
            raise DatasetError("A frozen arm failed the development score smoke check.")


def _load_test(raw: Path, snapshot: str) -> EllipticDataset:
    paths = [raw / name for name in EXPECTED_FILES]
    before = file_state(paths)
    if snapshot_digest({path.name: file_digest(path) for path in paths}) != snapshot:
        raise DatasetError("The source release does not match the pre-registration.")
    dataset = load_dataset(raw)
    if before != file_state(paths):
        raise DatasetError("A release file changed during the authorised read.")
    steps_seen = set(
        dataset.transactions.loc[
            dataset.transactions["time_step"].between(*TEST_STEPS), "time_step"
        ]
    )
    if steps_seen != set(range(TEST_STEPS[0], TEST_STEPS[1] + 1)):
        raise DatasetError("The release lacks the complete registered holdout window.")
    return dataset


def _maybe_publish(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".partial")
    write_json(temporary, record)
    temporary.replace(path)


def evaluate_once(
    root: Path, raw: Path, freeze: dict, arms: dict[str, Arm], *, freeze_sha: str
) -> Path:
    """Claim the single access before reading the release, even on failure."""
    if list(arms) != list(ARM_NAMES) or any(
        arms[key].feature_set is not (S1_TIER if key.startswith("s1") else S2_TIER)
        for key in arms
    ):
        raise DatasetError(
            "The prepared arms do not match the registered feature routing."
        )
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "holdout-attempt.json"
    access = {
        "status": "started",
        "started_at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        "git_commit": freeze["git_commit"],
        "source_snapshot_sha256": freeze["source_snapshot_sha256"],
        "approval_commit": freeze["approval_commit"],
        "freeze_sha256": freeze_sha,
        "arms": freeze["arms"],
        "steps": TEST_STEPS,
        "purpose": "Single pre-registered evaluation of the frozen four-arm shortlist",
    }
    try:
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(access, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise DatasetError(
            "A Phase 5 holdout attempt already exists. Stop and consult the ledger."
        ) from None
    stage = "release verification and ingestion"
    try:
        dataset = _load_test(raw, access["source_snapshot_sha256"])
        stage = "feature construction and prediction"
        features = {
            tier.scenario: build_feature_set(dataset, tier, *TEST_STEPS)
            for tier in (S1_TIER, S2_TIER)
        }
        if not features["S1"][1].equals(features["S2"][1]):
            raise DatasetError("The test feature windows have different eligible rows.")
        predictions = {}
        for key, arm in arms.items():
            frame, steps = features[arm.feature_set.scenario]
            predictions[key] = arm.score(frame)
        labels = dataset.transactions.set_index("tx_id")["class"].loc[steps.index]
        stage = "registered metrics and output publication"
        indices = step_draws(TEST_STEPS[1] - TEST_STEPS[0] + 1)
        metrics = {}
        for key, arm in arms.items():
            frame, steps = features[arm.feature_set.scenario]
            window = Window(frame, labels, steps, *TEST_STEPS)
            metrics[key] = final_metrics(
                window, predictions[key], arm.threshold, indices
            )
        report = {
            "access": {**access, "status": "completed"},
            "environment": freeze["environment"],
            "arms": metrics,
            "contrasts": paired_contrasts(metrics, indices),
        }
        with new_run(root, "evaluation") as (staging, destination):
            for key in ("s1_legacy", "s2_lgbm"):
                batch = staging / key
                batch.mkdir()
                ranked = ranked_batch_scores(
                    steps, predictions[key], SEED, 50, arms[key].threshold
                )
                ranked.to_csv(batch / "scores.csv")
                ranked.loc[ranked["in_review_queue"]].to_csv(batch / "review_queue.csv")
            write_json(staging / "evaluation.json", report)
            (staging / "evaluation.md").write_text(
                render_evaluation(report), encoding="utf-8"
            )
        _maybe_publish(
            receipt,
            {
                **access,
                "status": "completed",
                "output": destination.name,
                "evaluation_sha256": file_digest(destination / "evaluation.json"),
                "report_sha256": file_digest(destination / "evaluation.md"),
            },
        )
        return destination
    except BaseException as error:
        _maybe_publish(
            receipt,
            {
                **access,
                "status": "failed",
                "stage": stage,
                "error": type(error).__name__,
            },
        )
        raise DatasetError(
            f"Holdout attempt failed during {stage} ({type(error).__name__}). Do not retry."
        ) from None


def _track() -> None:
    receipt = read_json(ROOT / "holdout-attempt.json")
    if receipt["status"] != "completed":
        raise DatasetError("Tracking requires a completed holdout attempt.")
    source = ROOT / receipt["output"] / "evaluation.json"
    if file_digest(source) != receipt["evaluation_sha256"]:
        raise DatasetError("The completed aggregate output has a different digest.")
    result = read_json(source)
    for key, identity in receipt["arms"].items():
        scenario = key[:2].upper()
        config = {
            "scenario": scenario,
            "family": key,
            "fold": {"name": "holdout-40-49"},
            "model_sha256": identity["model_sha256"],
            "threshold": identity["threshold"],
            "source_snapshot_sha256": receipt["source_snapshot_sha256"],
            "freeze_sha256": receipt["freeze_sha256"],
            "availability_assumption": "conditional on A1"
            if scenario == "S2"
            else None,
        }
        row = {
            "config": config,
            "config_sha256": snapshot_digest(config),
            "git_commit": receipt["git_commit"],
            "validation": result["arms"][key],
        }
        run = record_fold_mlflow(
            ROOT, row, experiment_name=f"Elliptic {scenario} final evaluation"
        )
        print(f"Tracked {key} as {run}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run only the explicit Phase 5 command selected by the operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "prepare", help="fit and freeze from the verified development export"
    )
    sub.add_parser("track", help="import only the completed aggregate evaluation")
    for command in ("verify", "evaluate"):
        route = sub.add_parser(command)
        route.add_argument("--frozen", required=True, type=Path)
        route.add_argument("--freeze-sha", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = _prepare()
            print(f"Frozen shortlist: {path}", flush=True)
            print(f"Freeze SHA-256: {file_digest(path / 'freeze.json')}", flush=True)
        elif args.command == "track":
            _track()
        else:
            freeze, arms = _verify_frozen(args.frozen, args.freeze_sha)
            if args.command == "evaluate":
                path = evaluate_once(
                    ROOT, RAW, freeze, arms, freeze_sha=args.freeze_sha
                )
                print(f"Completed holdout evaluation: {path}", flush=True)
            else:
                _smoke_development(arms)
                print(
                    "Verified the frozen shortlist and executable revision.", flush=True
                )
    except (DatasetError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
