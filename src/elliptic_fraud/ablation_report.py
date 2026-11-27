"""Summarize development arms and apply the registered paired decision rule."""

import statistics
from collections.abc import Iterable


def _mean(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return statistics.mean(defined) if defined else None


def summarize_arm(scenario: str, name: str, family: str, runs: list[dict]) -> dict:
    """Summarize one arm with equal fold weights and operational step aggregates."""
    validations = [run["validation"] for run in runs]
    ap = [validation["mean_step_ap"] for validation in validations]
    defined = [value for value in ap if value is not None]
    queues = [validation["q_all"]["50"] for validation in validations]
    step_queues = [
        step
        for queue in queues
        for step in queue["by_step"].values()
        if step["eligible"]
    ]
    precision = _mean(
        step["precision"]
        for validation in validations
        for step in validation["q_lab"]["50"]["by_step"].values()
    )
    hits = sum(queue["confirmed_illicit"] for queue in queues)
    expected = sum(queue["expected_random_hits"] for queue in queues)
    return {
        "scenario": scenario,
        "features": name,
        "family": family,
        "fold_ap": ap,
        "mean_step_ap": _mean(ap),
        "fold_std": statistics.stdev(defined) if len(defined) > 1 else None,
        "p_lab_50": precision,
        "queue_vs_random": hits / expected if expected else None,
        "mean_tie_mass": _mean(step["tied_at_cut"] for step in step_queues),
        "small_ties": sum(step["tied_at_cut"] <= 5 for step in step_queues),
        "validation_steps": len(step_queues),
        "complexity": [
            max(len(run["columns"]) for run in runs),
            runs[0]["config"]["params"]["n_estimators"],
        ],
        "folds": [
            {
                "name": run["config"]["fold"]["name"],
                "config_sha256": run["config_sha256"],
                "mlflow_run_id": run["mlflow_run_id"],
            }
            for run in runs
        ],
    }


def _absolute_ties_ok(arm: dict) -> bool:
    steps = arm["validation_steps"]
    return steps > 0 and 10 * arm["small_ties"] >= 9 * steps


def paired_comparison(
    candidate: dict, baseline: dict, *, original: bool = False
) -> dict:
    """Compare matching fold APs and report the operational constraints."""
    deltas = [
        new - old if new is not None and old is not None else None
        for new, old in zip(candidate["fold_ap"], baseline["fold_ap"], strict=True)
    ]
    defined = [delta for delta in deltas if delta is not None]
    mean = _mean(defined)
    se = statistics.stdev(defined) / len(defined) ** 0.5 if len(defined) > 1 else None
    positive = sum(delta > 0 for delta in defined)
    precision, required = candidate["p_lab_50"], baseline["p_lab_50"]
    precision_ok = (
        precision is not None and required is not None and precision >= required
    )
    candidate_ties, baseline_ties = (
        candidate["mean_tie_mass"],
        baseline["mean_tie_mass"],
    )
    ties_ok = (
        _absolute_ties_ok(candidate)
        if original
        else candidate_ties is not None
        and baseline_ties is not None
        and candidate_ties <= baseline_ties
    )
    complete = len(defined) == len(deltas) == 4
    improvement = (
        complete and positive >= 3 and mean is not None and se is not None and mean > se
    )
    simpler = (
        complete
        and mean is not None
        and se is not None
        and abs(mean) <= se
        and (
            candidate["complexity"] < baseline["complexity"]
            if original
            else candidate["complexity"][0] < baseline["complexity"][0]
        )
    )
    return {
        "reference": {"features": baseline["features"], "family": baseline["family"]},
        "deltas": deltas,
        "delta": mean,
        "paired_se": se,
        "positive_folds": positive,
        "precision_ok": precision_ok,
        "ties_ok": ties_ok,
        "decision": (
            "improves"
            if improvement and precision_ok and ties_ok
            else "simpler within 1 SE"
            if simpler and precision_ok and ties_ok
            else "retained incumbent"
        ),
    }


def select_incumbent(
    arms: list[dict], *, original: bool = False
) -> tuple[dict, list[dict]]:
    """Apply the decision rule in declared order from the first XGBoost arm."""
    incumbent = arms[0]
    comparisons = []
    for arm in arms[1:]:
        comparison = paired_comparison(arm, incumbent, original=original)
        comparisons.append(
            {"features": arm["features"], "family": arm["family"], **comparison}
        )
        if comparison["decision"] != "retained incumbent":
            incumbent = arm
    return incumbent, comparisons


def review_selection(arms: list[dict]) -> dict:
    """Use the declared tier comparisons to review the next-phase feature sets."""
    decisions = {}
    for scenario in ("S1", "S2"):
        scenario_arms = [arm for arm in arms if arm["scenario"] == scenario]
        plan = {(arm["features"], arm["family"]): arm for arm in scenario_arms}
        comparisons = []
        eligible = []
        for arm in scenario_arms:
            pair = arm["paired_tier"]
            if pair is None:
                # Compare the reference tier's second family to its fixed XGBoost anchor.
                reference = scenario_arms[0]
                if arm is reference:
                    continue
            else:
                previous = pair["reference"]
                reference = plan[previous["features"], previous["family"]]
            comparison = paired_comparison(arm, reference)
            comparisons.append(
                {
                    "features": arm["features"],
                    "family": arm["family"],
                    "columns": arm["complexity"][0],
                    **comparison,
                }
            )
            if comparison["decision"] != "retained incumbent":
                eligible.append(arm)
        # Identify the strongest supported AP, then apply the paired simplicity rule.
        leader = max(
            eligible, key=lambda arm: arm["mean_step_ap"], default=scenario_arms[0]
        )
        simpler = [
            arm
            for arm in eligible
            if paired_comparison(arm, leader)["decision"] == "simpler within 1 SE"
        ]
        winner = min(
            simpler,
            key=lambda arm: (arm["complexity"][0], -arm["mean_step_ap"]),
            default=leader,
        )
        sequential, _ = select_incumbent(scenario_arms)
        short_list = [winner["features"]]
        if scenario == "S2":
            alternative = "T1b+T2+T2agg+T3"
            if alternative != winner["features"] and any(
                arm["features"] == alternative for arm in scenario_arms
            ):
                short_list.append(alternative)
        decisions[scenario] = {
            "winner": winner,
            "final_ties_ok": _absolute_ties_ok(winner),
            "sequential_winner": sequential,
            "comparisons": comparisons,
            "phase4": short_list,
        }
    return decisions


def _number(value: float | None, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if value is not None else "undefined"


def _render_decision_review(decisions: dict) -> list[str]:
    lines = [
        "## Decision rule review",
        "",
        "The original decisions above required an absolute cutoff-tie pass for every challenger. The initial reference could remain even if it failed that same requirement. This asymmetry rejected candidates that improved AP and reduced ties. The review identified the issue using the saved development fold aggregates and introduced no holdout access.",
        "",
        "The corrected comparisons require AP gains in at least three folds and a mean gain greater than one paired SE. A challenger must also meet the incumbent's labelled P@50 and have at most the incumbent's mean cutoff tie mass. Within one paired SE, fewer columns are preferred subject to the precision and relative tie constraints. The absolute cutoff requirement (at most five ties in at least 90% of steps) applies at final pre-registration or deployment.",
        "",
        "The corrected tier decisions use the reference paths declared in the configuration, with the same family on each tier edge. This compares a tier with the inputs it replaces or extends. Among arms that pass these comparisons, the highest mean AP is the reference for the paired simplicity check. The running-incumbent replay is reported separately because comparing across feature branches can make the result depend on declaration order.",
        "",
        "| Scenario | Candidate | Declared reference | Columns | Positive folds | Δ AP ± SE | Precision floor | Mean ties | Outcome |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for scenario, decision in decisions.items():
        for comparison in decision["comparisons"]:
            ref = comparison["reference"]
            lines.append(
                f"| {scenario} | {comparison['features']} ({comparison['family']}) | "
                f"{ref['features']} ({ref['family']}) | {comparison['columns']} | "
                f"{comparison['positive_folds']}/4 | "
                f"{_number(comparison['delta'])} ± {_number(comparison['paired_se'])} | "
                f"{'pass' if comparison['precision_ok'] else 'fail'} | "
                f"{'pass' if comparison['ties_ok'] else 'fail'} | {comparison['decision']} |"
            )
    lines.extend(["", "### Phase 4 candidates", ""])
    for scenario, decision in decisions.items():
        winner = decision["winner"]
        sequential = decision["sequential_winner"]
        candidates = ", ".join(f"`{name}`" for name in decision["phase4"])
        lines.append(
            f"- **{scenario}** carries forward {candidates}. The corrected choice is "
            f"{winner['features']} ({winner['family']}) with mean step AP "
            f"{_number(winner['mean_step_ap'])}. Its absolute tie check is "
            f"{winner['small_ties']}/{winner['validation_steps']} steps "
            f"({'pass' if decision['final_ties_ok'] else 'fail'}). A single running "
            f"incumbent under the corrected comparisons would retain {sequential['features']} "
            f"({sequential['family']})."
        )
    lines.extend(
        [
            "",
            "The A1 sensitivity above records the original experiment on the 543-column arm. This review reuses the completed folds, including that original sensitivity.",
            "",
        ]
    )
    return lines


def render_ablation(summary: dict) -> str:
    """Render a reviewable Markdown report entirely from aggregate results."""
    lines = [
        "# Phase 3 feature-tier ablation",
        "",
        f"Development frame SHA-256: `{summary['development_snapshot_sha256']}`.",
        f"Original CSV snapshot SHA-256: `{summary['source_snapshot_sha256']}`.",
        f"Experiment implementation revision: `{summary['git_commit']}`.",
        f"Python: {summary['environment']['python']}. Packages: "
        + ", ".join(
            f"{name} {version}"
            for name, version in summary["environment"]["packages"].items()
        )
        + ".",
        "",
        "The experiment uses the four expanding development folds and the phase 3 configuration in `configs/phase3-ablation.json`. XGBoost uses the legacy configuration, including its single-thread setting. Random forest fits 500 trees with minimum leaf size 2 and six workers, then scores with one worker to fix floating-point accumulation order. Both use seed 42, a descriptive threshold of 0.5, and no early stopping or embargo. Each fold fits only known labels. The initial references are T0 XGBoost for S1 and T1b XGBoost for S2. S1 uses only ancestry. **All S2 results are conditional on A1.**",
        "",
        "From a clean checkout with the existing verified export in `data/processed/development/`, run `elliptic-experiment --config configs/phase3-ablation.json`. The ablation reads only the export.",
        "",
        "AP is averaged over validation steps with known illicit cases and then over folds. The table shows sample SD across folds and the paired AP difference against the declared tier with the same family. Labelled P@50 is averaged over steps where it is defined. Queue lift is summed known hits divided by summed uniform random expectations at 50 slots per step. Tie mass is the number of transactions tied at the cutoff.",
        "",
        "| Scenario | Feature set | Family | Mean step AP ± SD | Paired tier reference | Δ AP ± SE | Labelled P@50 | Queue lift | Mean cutoff ties | Steps with ≤5 cutoff ties |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    arms = summary["arms"]
    if summary["sensitivity"]:
        arms = [*arms, summary["sensitivity"]["filtered"]]
    for arm in arms:
        pair = arm["paired_tier"]
        lines.append(
            f"| {arm['scenario']} | {arm['features']} | {arm['family']} | "
            f"{_number(arm['mean_step_ap'])} ± {_number(arm['fold_std'])} | "
            f"{pair['reference']['features'] if pair else '—'} | "
            f"{_number(pair['delta']) + ' ± ' + _number(pair['paired_se']) if pair else '—'} | "
            f"{_number(arm['p_lab_50'])} | {_number(arm['queue_vs_random'])} | "
            f"{_number(arm['mean_tie_mass'], 2)} | {arm['small_ties']}/{arm['validation_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Original decisions",
            "",
            "Arms are considered in configuration order, with XGBoost before random forest. A candidate must improve AP in at least three folds, exceed one paired SE on average, match the incumbent's labelled P@50 and have at most five tied transactions in at least 90% of steps. Within one paired SE, prefer fewer columns and then fewer trees when both operational constraints pass.",
            "",
        ]
    )
    for scenario, decision in summary["selection"].items():
        winner = decision["incumbent"]
        status = (
            "meets the cutoff-tie constraint"
            if decision["ties_ok"]
            else "remains a reference because the cutoff-tie constraint was unmet"
        )
        lines.append(
            f"- **{scenario}** retains {winner['features']} ({winner['family']}), which {status}."
        )
    lines.extend(
        [
            "",
            "| Scenario | Candidate | Incumbent at comparison | Positive folds | Δ AP ± SE | Precision floor | Tie constraint | Outcome |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for scenario, decision in summary["selection"].items():
        for comparison in decision["comparisons"]:
            ref = comparison["reference"]
            lines.append(
                f"| {scenario} | {comparison['features']} ({comparison['family']}) | "
                f"{ref['features']} ({ref['family']}) | {comparison['positive_folds']}/4 | "
                f"{_number(comparison['delta'])} ± {_number(comparison['paired_se'])} | "
                f"{'pass' if comparison['precision_ok'] else 'fail'} | "
                f"{'pass' if comparison['ties_ok'] else 'fail'} | {comparison['decision']} |"
            )
    lines.extend(
        [
            "",
            "## A1 sensitivity",
            "",
            "Within each fold, partial Spearman is computed on training rows for each T2/T3 column against out-degree, controlling for in-degree. The top 5% of defined absolute coefficients are flagged, rounded up. The strongest S2 arm under the decision rule is repeated when it contains flagged inputs. When the structural reference survives, the highest-AP S2 arm with applicable flags is used for this diagnostic. Removing a flagged T2 column also removes its matching parent and grandparent aggregates. The probe cannot establish feature availability, and this diagnostic does not change the tier decision.",
            "",
            "| Fold | Training steps | Defined columns | Flagged | Removed predictors |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for fold, probe in summary["probe"].items():
        start, end = probe["train"]
        removed = (
            summary["sensitivity"]["removed"][fold] if summary["sensitivity"] else 0
        )
        lines.append(
            f"| {fold} | {start}–{end} | "
            f"{sum(value is not None for value in probe['partial_spearman'].values())} | "
            f"{len(probe['flagged'])} | {removed} |"
        )
    if summary["sensitivity"]:
        source = summary["sensitivity"]["source"]
        lines.extend(
            [
                "",
                f"The filtered row repeats **{source['features']} ({source['family']})**.",
            ]
        )
    lines.extend(["", f"Aggregate artifact: `{summary['artifact']}`.", ""])
    lines.extend(_render_decision_review(review_selection(summary["arms"])))
    return "\n".join(lines)
