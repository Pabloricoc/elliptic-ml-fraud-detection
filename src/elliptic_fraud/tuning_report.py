"""Render the Phase 4 report from aggregate artifacts only."""


def _number(value: float | None, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if value is not None else "undefined"


def _description(arm: dict) -> str:
    return f"{arm['features']} · {arm['family']} · {arm['stage']} (`{arm['id'][:10]}`)"


def _passed(value: bool) -> str:
    return "pass" if value else "fail"


def render_tuning(summary: dict) -> str:
    """List every completed arm, trial disposition, sensitivity and next decision."""
    plan, counts = summary["plan"], summary["counts"]
    attempts = [
        attempt for study in summary["studies"] for attempt in study["attempts"]
    ]
    lines = [
        "# Phase 4 tuned boosters and ensembles",
        "",
        f"Development frames: `{summary['development_snapshot_sha256']}`.",
        f"Source snapshot: `{summary['source_snapshot_sha256']}`.",
        f"Experiment revision: `{summary['git_commit']}`.",
        f"Python {summary['environment']['python']}. "
        + ", ".join(
            f"{name} {version}"
            for name, version in summary["environment"]["packages"].items()
        )
        + ".",
        "",
        "The experiment reads the verified export of steps 1–39 and the saved Phase 3 reference metrics. All S2 results are **conditional on A1**. S1 uses T0 and a bounded T1 attempt to reduce ties. The API remains on the approved legacy model.",
        "",
        f"Reproduce from a clean checkout with `elliptic-tune --config configs/phase4-tuning.json --phase3-results artifacts/experiments/{summary['phase3_reference']}`. Results resume from local Optuna and fold caches.",
        "",
        f"The plan allows {plan['trials_per_tier']} TPE trials per tier and family, at most {2 * plan['trials_per_tier']} per family and scenario, and {plan['max_rounds']} boosting rounds. TPE uses seed 42. Median pruning uses cumulative mean step AP after each fold once two trials complete and two folds have been observed. The family/scenario search stops after 30 attempts without a cumulative +0.003 gain. The optional pairwise ranking objective was outside this budget.",
        "",
        "Each outer fold reserves the final three steps of training to choose the round count by log loss with patience 100. A fresh booster then fits the full outer training window for that count. Class weights use known labels from the rows fitted at each stage. Each best trial is also scored using five seeds (42–46) and the round count chosen by seed 42 in the corresponding fold. The remaining seeds refit only the outer training window.",
        "",
        f"Optuna attempted **{len(attempts)} trials**, including **{sum(row['state'] == 'COMPLETE' for row in attempts)} complete** and **{sum(row['state'] == 'PRUNED' for row in attempts)} pruned**. Across trials, seed replicas, blends and embargo, there were **{counts['evaluated_configurations']} evaluated configurations** and **{counts['fold_reports']} saved fold reports**. Configuration counts group identical parameters, seeds and stages across folds. A pruned trial may have saved a subset of folds.",
        "",
        "## Reference arms",
        "",
        "| Scenario | Feature set | Family | Mean step AP | Labelled P@50 | Mean cutoff ties | Steps with ≤5 ties |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario, arm in summary["anchors"].items():
        lines.append(
            f"| {scenario} | {arm['features']} | {arm['family']} | {_number(arm['mean_step_ap'])} | "
            f"{_number(arm['p_lab_50'])} | {_number(arm['mean_tie_mass'], 2)} | {arm['small_ties']}/{arm['validation_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Walk-forward results",
            "",
            "AP is the equally weighted mean across F1–F4 of per-step illicit AP on known labels. The comparison uses the Phase 3 rule with the fixed reference for each scenario. Gains must be positive in at least three folds and exceed one paired SE. Labelled P@50 must meet its floor and mean cutoff ties must be at or below the reference. Within one paired SE, fewer columns are preferred under those same operational constraints. A blend must also pass the rule against its parent. The absolute tie requirement applies at final pre-registration or deployment. S1 candidates must pass it to advance from this phase.",
            "",
            "| Scenario | Arm | Step AP ± fold SD | Δ AP ± paired SE | Positive folds | P@50 | Queue lift | Mean ties | Steps with ≤5 ties | Relative decision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for arm in summary["arms"]:
        pair = arm["vs_reference"]
        status = (
            "passes"
            if arm["retained"]
            else "queue-only diagnostic"
            if arm.get("queue_only")
            else "reference retained"
        )
        lines.append(
            f"| {arm['scenario']} | {_description(arm)} | {_number(arm['mean_step_ap'])} ± {_number(arm['fold_std'])} | "
            f"{_number(pair['delta'])} ± {_number(pair['paired_se'])} | {pair['positive_folds']}/4 | "
            f"{_number(arm['p_lab_50'])} | {_number(arm['queue_vs_random'])} | {_number(arm['mean_tie_mass'], 2)} | "
            f"{arm['small_ties']}/{arm['validation_steps']} | {status} |"
        )
    lines.extend(
        [
            "",
            "### Search dispositions and selected rounds",
            "",
            "| Scenario | Tier | Family | Trial states in order | Best trial ID | Best rounds by fold |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    arms_by_id = {arm["id"]: arm for arm in summary["arms"]}
    for study in summary["studies"]:
        states = ", ".join(
            f"{row['number']} {row['state']} ({row['observed_folds']}/4)"
            for row in study["attempts"]
        )
        rounds = arms_by_id[study["best"]]["fold_rounds"]
        lines.append(
            f"| {study['scenario']} | {study['tier']} | {study['family']} | {states} | `{study['best'][:10]}` | {', '.join(map(str, rounds))} |"
        )
    lines.extend(["", "## Rank averaging", ""])
    lines.append(
        "Ranks use average ties and percentiles over all eligible nodes within each step. Pairwise Spearman is averaged over steps with defined correlations. A pair above 0.97 or an undefined average stops the ensemble. If the two boosters pass, the matching Phase 3 forest is replayed on development and checked against its saved fold AP. Rank scores describe a review queue. Their scale is ordinal, so the stored probability-style diagnostics have no calibration interpretation. A within-step rank uses the completed batch and is only a queue diagnostic for S1."
    )
    lines.append("")
    for scenario, agreement in summary["rank_ensembles"].items():
        pairs = ", ".join(
            f"{name} {_number(value)}" for name, value in agreement["pairs"].items()
        )
        result = (
            "evaluated" if agreement["evaluated"] else "stopped at the diversity check"
        )
        lines.append(f"- **{scenario}** {result}. Mean Spearman: {pairs}.")
    lines.extend(
        [
            "",
            "## Two-step embargo sensitivity",
            "",
            "This diagnostic repeats the highest-AP configuration in S1 and the two highest-AP configurations in S2 from the per-study bests. A seed average replaces its parent for this ranking only when it meets the relative rule. The diagnostic recomputes inner stopping with shorter training ranges and keeps the same validation steps. The rank ensemble is outside this sensitivity.",
            "",
            "| Scenario | Arm | Step AP | Δ vs g=0 ± SE | P@50 | Mean ties | Steps with ≤5 ties |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in summary["embargo"]:
        pair = arm["vs_no_embargo"]
        lines.append(
            f"| {arm['scenario']} | {_description(arm)} | {_number(arm['mean_step_ap'])} | "
            f"{_number(pair['delta'])} ± {_number(pair['paired_se'])} | {_number(arm['p_lab_50'])} | "
            f"{_number(arm['mean_tie_mass'], 2)} | {arm['small_ties']}/{arm['validation_steps']} |"
        )
    lines.extend(["", "## Decisions", ""])
    absolute = [
        arm
        for arm in [*summary["arms"], *summary["embargo"]]
        if arm["scenario"] == "S1"
        and arm["absolute_ties_ok"]
        and not arm.get("queue_only")
    ]
    if not absolute:
        lines.append(
            "**S1 negative result.** No evaluated confirmation-time arm reached at most five cutoff ties in at least 90% of validation steps. The legacy T0 XGBoost remains the API model."
        )
    elif summary["selected"]["S1"] is None:
        lines.append(
            "Some S1 arms met the absolute tie requirement. The Phase 3 reference stays because the measured alternatives did not clear both the relative comparison and the final tie requirement."
        )
    else:
        lines.append(
            f"S1 development candidate: {_description(summary['selected']['S1'])}. The API remains on the legacy model pending the final evaluation and explicit approval."
        )
    selected = summary["selected"]["S2"]
    if selected is None:
        lines.append(
            "S2 retains the Phase 3 reference for development comparisons. All reported S2 outcomes remain conditional on A1."
        )
    else:
        lines.append(
            f"S2 development candidate, conditional on A1: {_description(selected)}. Its absolute tie check covers {selected['small_ties']}/{selected['validation_steps']} steps ({_passed(selected['absolute_ties_ok'])})."
        )
    lines.extend(
        [
            "",
            "The full local aggregate retains trial parameters, per-fold rounds, paired gate outcomes and fold run digests.",
            "",
        ]
    )
    return "\n".join(lines)
