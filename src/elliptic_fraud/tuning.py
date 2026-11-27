"""Tune and compare development-only boosters under the reviewed Phase 3 rules."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock
from optuna import logging as optuna_logging

from .ablation import load_cached_ablation
from .ablation_report import _absolute_ties_ok, paired_comparison
from .development import DEVELOPMENT_DIR, load_development_export
from .evaluate import evaluation_environment
from .experiments import clean_git_revision
from .folds import development_folds
from .runs import implementation_digest, read_json, snapshot_digest, write_json
from .schema import DatasetError
from .tuned_models import FAMILIES
from .tuning_ensemble import rank_ensemble
from .tuning_report import render_tuning
from .tuning_runs import BoosterArm, TuningRun
from .tuning_search import (
    Search,
    plateau_reached,
    score_booster,
    seed_average,
    tune_tier,
)

TIERS = {
    "S1": ["T0", "T1"],
    "S2": ["T1b+T2+T3", "T1b+T2+T2agg+T3"],
}
ANCHORS = {"S1": ("T0", "xgboost"), "S2": ("T1b+T2+T3", "random_forest")}


def _limits(document: dict) -> tuple[int, int]:
    trials, rounds = document.get("trials_per_tier"), document.get("max_rounds")
    if document.get("tiers") != TIERS:
        raise DatasetError(
            "The tuning plan must use the approved S1 and S2 feature sets."
        )
    if type(trials) is not int or not 1 <= trials <= 30:
        raise DatasetError(
            "Two tiers may use at most 60 trials per family and scenario."
        )
    if type(rounds) is not int or not 1 <= rounds <= 2000:
        raise DatasetError("The boosting ceiling must be between 1 and 2000.")
    return trials, rounds


def _reference(summary: dict, scenario: str, tier: str, family: str) -> dict:
    for arm in summary["arms"]:
        if (arm["scenario"], arm["features"], arm["family"]) == (
            scenario,
            tier,
            family,
        ):
            return arm
    raise DatasetError("The verified Phase 3 results lack a required reference arm.")


def _arm_key(arm: BoosterArm) -> str:
    return snapshot_digest(
        {"folds": [row.report["config_sha256"] for row in arm.folds]}
    )


def _advanced(comparison: dict) -> bool:
    return comparison["decision"] != "retained incumbent"


def _measured(
    arm: BoosterArm, baseline: dict, parent: BoosterArm | None = None
) -> dict:
    aggregate = arm.aggregate()
    pair = paired_comparison(aggregate, baseline)
    stage_pair = (
        paired_comparison(aggregate, parent.aggregate()) if parent is not None else None
    )
    return {
        **aggregate,
        "id": _arm_key(arm),
        "vs_reference": pair,
        "vs_parent": stage_pair,
        "absolute_ties_ok": _absolute_ties_ok(aggregate),
        "retained": _advanced(pair) and (stage_pair is None or _advanced(stage_pair)),
    }


def _best(arms: list[BoosterArm]) -> BoosterArm:
    return max(arms, key=lambda arm: arm.aggregate()["mean_step_ap"])


def _embargo(
    run: TuningRun,
    candidates: list[tuple[Search, BoosterArm, BoosterArm]],
) -> list[dict]:
    """Recompute inner stopping and any selected seed averaging under g=2."""
    chosen = []
    for scenario, count in (("S1", 1), ("S2", 2)):
        eligible = [
            candidate for candidate in candidates if candidate[0].scenario == scenario
        ]
        chosen.extend(
            sorted(eligible, key=lambda item: -item[2].aggregate()["mean_step_ap"])[
                :count
            ]
        )
    folds = development_folds(2)
    results = []
    for search, base, selected in chosen:
        window = run.window(search.scenario, search.tier)
        rows = [score_booster(run, window, search, fold, base.params) for fold in folds]
        replay = BoosterArm(
            search.scenario, search.tier, search.family, "TPE", base.params, rows
        )
        if selected.stage == "5 seeds":
            replay = seed_average(run, window, search, replay, folds)
        aggregate = replay.aggregate()
        results.append(
            {
                **aggregate,
                "id": _arm_key(replay),
                "source": _arm_key(selected),
                "vs_no_embargo": paired_comparison(aggregate, selected.aggregate()),
                "absolute_ties_ok": _absolute_ties_ok(aggregate),
            }
        )
        print(
            f"Completed g=2 {search.scenario} {search.tier} {search.family} ({selected.stage})",
            flush=True,
        )
    return results


def run_tuning(
    config_path: Path,
    development_dir: Path,
    phase3_results: Path,
    output: Path,
    *,
    git_commit: str,
) -> Path:
    """Compare bounded tuning, five-seed averages, ranks and the embargo diagnostic."""
    document = read_json(config_path)
    trial_limit, max_rounds = _limits(document)
    output = output.resolve()
    cache_root = (development_dir.resolve().parent / "phase4-features").resolve()
    protected = [
        development_dir.resolve(),
        Path("data/raw").resolve(),
        *(
            (Path("artifacts") / name).resolve()
            for name in ("models", "scores", "eda", "tracking")
        ),
    ]
    if any(
        path.is_relative_to(root) for path in (output, cache_root) for root in protected
    ):
        raise DatasetError(
            "Choose tuning locations outside the export, raw data and protected artifacts."
        )
    phase3 = load_cached_ablation(phase3_results)
    dataset, manifest = load_development_export(development_dir)
    fingerprints = {
        name: manifest[name]
        for name in ("development_snapshot_sha256", "source_snapshot_sha256")
    }
    if any(phase3[name] != value for name, value in fingerprints.items()):
        raise DatasetError(
            "The Phase 3 reference and tuning export must identify the same snapshot."
        )
    environment = evaluation_environment()
    environment["packages"].update(
        {name: version(name) for name in ("lightgbm", "optuna")}
    )
    provenance = {
        **fingerprints,
        "implementation_sha256": implementation_digest(),
        "environment": environment,
        "protocol_sha256": snapshot_digest(document),
    }
    identity = snapshot_digest({**provenance, "phase3": phase3_results.name})
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"phase4-{identity}.json"
    run = TuningRun(dataset, cache_root, output, provenance, git_commit)
    anchors = {
        scenario: _reference(phase3, scenario, *key)
        for scenario, key in ANCHORS.items()
    }
    optuna_logging.set_verbosity(optuna_logging.WARNING)
    with FileLock(output / f".phase4-{identity}.lock"):
        if not path.exists():
            summary = _experiments(
                run,
                document,
                anchors,
                phase3,
                trial_limit,
                max_rounds,
                phase3_results.name,
            )
            temporary = path.with_suffix(".partial")
            write_json(temporary, summary)
            temporary.replace(path)
        summary = read_json(path)
        markdown = path.with_suffix(".md")
        temporary_markdown = markdown.with_suffix(".md.partial")
        temporary_markdown.write_text(render_tuning(summary), encoding="utf-8")
        temporary_markdown.replace(markdown)
    return path


def _experiments(
    run: TuningRun,
    document: dict,
    anchors: dict,
    phase3: dict,
    trial_limit: int,
    max_rounds: int,
    reference_file: str,
) -> dict:
    storage = run.output / "optuna-phase4.db"
    folds = development_folds()
    trials, measured, bests = [], [], []
    history = {(scenario, family): [] for scenario in TIERS for family in FAMILIES}
    for scenario, tiers in TIERS.items():
        for tier in tiers:
            window = run.window(scenario, tier)
            for family in FAMILIES:
                prior = history[scenario, family]
                if plateau_reached(prior):
                    continue
                search = Search(scenario, tier, family, trial_limit, max_rounds)
                arms, best, attempted = tune_tier(run, storage, window, search, prior)
                prior.extend(attempted)
                trials.append(
                    {
                        "scenario": scenario,
                        "tier": tier,
                        "family": family,
                        "best": _arm_key(best),
                        "attempts": [
                            {
                                "number": trial.number + 1,
                                "state": trial.state.name,
                                "params": trial.params,
                                "objective": trial.value
                                if trial.state.name == "COMPLETE"
                                else None,
                                "observed_folds": len(trial.intermediate_values),
                            }
                            for trial in attempted
                        ],
                    }
                )
                measured.extend(_measured(arm, anchors[scenario]) for arm in arms)
                seeded = seed_average(run, window, search, best, folds)
                row = _measured(seeded, anchors[scenario], best)
                measured.append(row)
                bests.append((search, best, seeded if row["retained"] else best))
                print(
                    f"Completed five-seed average {scenario} {tier} {family}",
                    flush=True,
                )

    correlations = {}
    for scenario in TIERS:
        xgb, lgbm = (
            _best(
                [
                    arm
                    for search, _, arm in bests
                    if search.scenario == scenario and search.family == family
                ]
            )
            for family in FAMILIES
        )
        forest = _reference(phase3, scenario, TIERS[scenario][0], "random_forest")
        blended, agreement = rank_ensemble(run, xgb, lgbm, forest)
        correlations[scenario] = {
            "pairs": agreement,
            "evaluated": blended is not None,
            "members": [_arm_key(xgb), _arm_key(lgbm)],
        }
        if blended is not None:
            row = _measured(blended, anchors[scenario], _best([xgb, lgbm]))
            # A within-step percentile requires the complete step at scoring time.
            if scenario == "S1":
                row["retained"] = False
                row["queue_only"] = True
            measured.append(row)

    embargo = _embargo(run, bests)
    selected = {}
    for scenario in TIERS:
        eligible = [
            row for row in measured if row["scenario"] == scenario and row["retained"]
        ]
        if scenario == "S1":
            eligible = [row for row in eligible if row["absolute_ties_ok"]]
        if eligible:
            leader = max(eligible, key=lambda row: row["mean_step_ap"])
            simpler = [
                row
                for row in eligible
                if paired_comparison(row, leader)["decision"] == "simpler within 1 SE"
            ]
            selected[scenario] = min(
                simpler,
                key=lambda row: (row["complexity"][0], -row["mean_step_ap"]),
                default=leader,
            )
        else:
            selected[scenario] = None

    return {
        **run.provenance,
        "git_commit": run.git_commit,
        "phase3_reference": reference_file,
        "plan": document,
        "anchors": anchors,
        "studies": trials,
        "arms": measured,
        "rank_ensembles": correlations,
        "embargo": embargo,
        "selected": selected,
        "counts": _counts(run),
    }


def _counts(run: TuningRun) -> dict:
    configs = set()
    fold_count = 0
    for path in run.output.glob("*/evaluation.json"):
        saved = read_json(path)["config"]
        if saved.get("phase") != 4 or any(
            saved.get(key) != value for key, value in run.provenance.items()
        ):
            continue
        fold_count += 1
        comparison = {
            key: value
            for key, value in saved.items()
            if key
            not in {
                "fold",
                "inner_early_stopping",
                "fixed_rounds",
                "members",
                "member_digests",
                "params",
            }
        }
        comparison["params"] = {
            name: value
            for name, value in saved["params"].items()
            if name != "n_estimators"
        }
        configs.add(snapshot_digest(comparison))
    if len(configs) > 600:
        raise DatasetError("The development configuration budget was exceeded.")
    return {"fold_reports": fold_count, "evaluated_configurations": len(configs)}


def main(argv: list[str] | None = None) -> int:
    """Run Phase 4 from JSON, cached Phase 3 results and the development export."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase3-results", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, default=DEVELOPMENT_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args(argv)
    try:
        path = run_tuning(
            args.config,
            args.development_dir,
            args.phase3_results,
            args.output_dir,
            git_commit=clean_git_revision(),
        )
    except (OSError, DatasetError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        f"Aggregate tuning and report written to {path} and {path.with_suffix('.md')}."
    )
    return 0
