"""Run and resume the bounded per-tier Optuna searches."""

import statistics
from dataclasses import dataclass
from pathlib import Path

from optuna import Trial, TrialPruned, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from .folds import Fold, development_folds
from .model_features import Window
from .runs import snapshot_digest
from .schema import DatasetError
from .train import SEED
from .tuned_models import PATIENCE, SEEDS, fit_booster, suggest_parameters
from .tuning_runs import BoosterArm, FoldResult, TuningRun


@dataclass(frozen=True)
class Search:
    """The fixed routing and resource limits of a single study."""

    scenario: str
    tier: str
    family: str
    trials: int
    max_rounds: int


def plateau_reached(trials: list[FrozenTrial]) -> bool:
    """Stop after 30 attempts since the last cumulative gain of at least 0.003."""
    reference = float("-inf")
    stale = 0
    for trial in trials:
        if (
            trial.state == TrialState.COMPLETE
            and trial.value is not None
            and trial.value >= reference + 0.003
        ):
            reference = trial.value
            stale = 0
        else:
            stale += 1
    return stale >= 30


def score_booster(
    run: TuningRun,
    window: Window,
    search: Search,
    fold: Fold,
    params: dict,
    *,
    seed: int = SEED,
    rounds: int | None = None,
) -> FoldResult:
    """Measure or reuse exactly one parameter, seed and fold combination."""
    config = {
        "scenario": search.scenario,
        "feature_set": search.tier,
        "family": search.family,
        "params": {**params, "n_estimators": search.max_rounds},
        "seed": seed,
        "patience": PATIENCE,
        "fixed_rounds": rounds,
        "feature_version": 1,
        "prediction_jobs": 1,
    }
    return run.fold(
        window,
        fold,
        config,
        lambda train, validation: fit_booster(
            search.family,
            params,
            train,
            validation,
            fold,
            max_rounds=search.max_rounds,
            seed=seed,
            fixed_rounds=rounds,
        ),
    )


def tune_tier(
    run: TuningRun,
    storage: Path,
    window: Window,
    search: Search,
    previous: list[FrozenTrial],
) -> tuple[list[BoosterArm], BoosterArm, list[FrozenTrial]]:
    """Search a tier with TPE and a median pruner after each completed fold."""
    folds = development_folds()
    key = snapshot_digest({**run.provenance, "search": vars(search)})
    study = create_study(
        storage=f"sqlite:///{storage.as_posix()}",
        study_name=key,
        sampler=TPESampler(seed=SEED, n_startup_trials=2),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        direction="maximize",
        load_if_exists=True,
    )
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
    finished = {}

    def objective(trial: Trial) -> float:
        params = suggest_parameters(trial, search.family)
        results, values = [], []
        for step, fold in enumerate(folds):
            result = score_booster(run, window, search, fold, params)
            results.append(result)
            value = result.report["validation"]["mean_step_ap"]
            if value is None:
                raise TrialPruned("Undefined mean step AP.")
            values.append(value)
            trial.report(statistics.mean(values), step)
            if step + 1 < len(folds) and trial.should_prune():
                raise TrialPruned()
        finished[trial.number] = results
        return statistics.mean(values)

    while len(study.trials) < search.trials and not plateau_reached(
        [*previous, *study.trials]
    ):
        study.optimize(objective, n_trials=1)
        trial = study.trials[-1]
        print(
            f"{search.scenario} {search.tier} {search.family} trial {trial.number + 1}: {trial.state.name}",
            flush=True,
        )
    arms = {}
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        rows = finished.get(trial.number)
        if rows is None:
            rows = [
                score_booster(run, window, search, fold, trial.params) for fold in folds
            ]
        arms[trial.number] = BoosterArm(
            search.scenario, search.tier, search.family, "TPE", trial.params, rows
        )
    if not arms:
        raise DatasetError("The study needs at least one completed configuration.")
    return list(arms.values()), arms[study.best_trial.number], study.trials


def seed_average(
    run: TuningRun,
    window: Window,
    search: Search,
    selected: BoosterArm,
    folds: tuple[Fold, ...],
) -> BoosterArm:
    """Average the selected seed and four refits at its frozen per-fold rounds."""
    rows = []
    for fold, anchor in zip(folds, selected.folds, strict=True):
        rounds = anchor.report["best_rounds"]
        members = [
            anchor,
            *(
                score_booster(
                    run, window, search, fold, selected.params, seed=seed, rounds=rounds
                )
                for seed in SEEDS[1:]
            ),
        ]
        scores = sum(member.scores for member in members) / len(members)
        config = {
            "scenario": search.scenario,
            "feature_set": search.tier,
            "family": search.family,
            "params": {**selected.params, "n_estimators": rounds},
            "stage": "seed_average",
            "seeds": SEEDS,
            "members": [member.report["config_sha256"] for member in members],
        }
        rows.append(
            run.fold(
                window,
                fold,
                config,
                lambda *_, scores=scores, rounds=rounds: (scores, rounds),
            )
        )
    return BoosterArm(
        search.scenario, search.tier, search.family, "5 seeds", selected.params, rows
    )
