"""Aggregate and render the pre-registered Phase 5 analyses."""

from collections.abc import Iterable, Sequence
from statistics import mean

import numpy as np
import pandas as pd

from .evaluate import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    LABELLED_CAPACITIES,
    evaluate_window,
)
from .model_evaluation import block_bootstrap_ci
from .model_features import Window
from .phase5_models import ARM_NAMES, TEST_STEPS

COMPARATORS = {
    "s2_rf": ("Primary", "S2 LightGBM − S2 Random Forest"),
    "s1_legacy": ("Primary", "S2 LightGBM − S1 legacy XGBoost"),
    "s1_refit": ("Descriptive", "S2 LightGBM − S1 T0 XGBoost refit"),
}


def step_draws(steps: int) -> np.ndarray:
    """Generate the shared, seeded sample of whole step indices."""
    return np.random.default_rng(BOOTSTRAP_SEED).choice(
        steps, size=(BOOTSTRAP_DRAWS, steps)
    )


def _step_mean(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def final_metrics(
    window: Window, scores: pd.Series, threshold: float, indices: np.ndarray
) -> dict:
    """Measure one frozen score series and both descriptive step windows."""
    result = evaluate_window(
        window,
        scores,
        seed=BOOTSTRAP_SEED,
        threshold=threshold,
        bootstrap_indices=indices,
    )
    ap = result["per_step_ap"]
    result["defined_steps"] = sum(value is not None for value in ap.values())
    split = result["split_at_43"]
    split["before_43_mean_step_ap"] = _step_mean(ap[step] for step in range(40, 43))
    split["from_43_mean_step_ap"] = _step_mean(ap[step] for step in range(43, 50))
    rows = result["q_all"][50]["by_step"].values()
    mass = [row["tied_at_cut"] for row in rows if row["eligible"]]
    result["cutoff_ties"] = {
        "mean": _step_mean(mass),
        "small_ties": sum(value <= 5 for value in mass),
        "eligible_steps": len(mass),
    }
    return result


def _interpret(interval: dict, primary: bool) -> str:
    lower, upper = interval["lower"], interval["upper"]
    if lower is None or upper is None:
        return "Undefined interval."
    if lower > 0:
        return (
            "S2 LightGBM has an advantage in this period."
            if primary
            else "Descriptive interval entirely above zero."
        )
    if upper < 0:
        return (
            "S2 LightGBM has a disadvantage in this period."
            if primary
            else "Descriptive interval entirely below zero."
        )
    return (
        "A difference was not demonstrated in this period."
        if primary
        else "Descriptive interval includes zero."
    )


def paired_contrasts(results: dict, indices: np.ndarray) -> list[dict]:
    """Bootstrap paired AP differences with the same draw matrix as each arm."""
    candidate = results["s2_lgbm"]["per_step_ap"]
    contrasts = []
    for key, (role, label) in COMPARATORS.items():
        baseline = results[key]["per_step_ap"]
        effects = {}
        for step in candidate:
            left, right = candidate[step], baseline[step]
            effects[step] = (
                left - right if left is not None and right is not None else None
            )
        interval = block_bootstrap_ci(effects, indices=indices)
        contrasts.append(
            {
                "role": role,
                "label": label,
                "steps": sum(value is not None for value in effects.values()),
                **interval,
                "conclusion": _interpret(interval, role == "Primary"),
            }
        )
    return contrasts


def _number(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _interval(values: dict) -> str:
    return f"[{_number(values['lower'])}, {_number(values['upper'])}]"


def _table(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    def line(values: Sequence[object]) -> str:
        return "| " + " | ".join(str(value) for value in values) + " |"

    return "\n".join([line(columns), line(["---"] * len(columns)), *map(line, rows)])


def _summary_tables(results: dict) -> list[str]:
    return [
        "## Average precision and probability metrics",
        _table(
            (
                "Arm",
                "Mean step AP",
                "95% CI",
                "Defined steps",
                "Valid draws",
                "Step AP SD",
                "Pooled AP",
                "ROC-AUC",
                "Brier",
            ),
            (
                (
                    ARM_NAMES[key],
                    _number(row["mean_step_ap"]),
                    _interval(row["mean_step_ap_ci"]),
                    row["defined_steps"],
                    row["mean_step_ap_ci"]["defined_draws"],
                    _number(row["step_ap_std"]),
                    _number(row["pooled_ap"]),
                    _number(row["roc_auc"]),
                    _number(row["brier_score"]),
                )
                for key, row in results.items()
            ),
        ),
        "## Labelled queue precision and recall",
        _table(
            ("Arm", "K", "Mean step precision", "95% precision CI", "Mean step recall"),
            (
                (
                    ARM_NAMES[key],
                    cap,
                    _number(queue["mean_step_precision"]),
                    _interval(queue["precision_ci"]),
                    _number(queue["mean_step_recall"]),
                )
                for key, row in results.items()
                for cap, queue in row["q_lab"].items()
            ),
        ),
        "## All-transaction queue and uniform random baseline",
        "K=50 is the registered operating capacity. K=25 and K=100 are supporting diagnostics.",
        _table(
            (
                "Arm",
                "K",
                "Reviewed",
                "Illicit",
                "Licit",
                "Unlabelled",
                "Unlabelled share",
                "Known-illicit recall",
                "Expected random hits",
                "Observed / expected",
            ),
            (
                (
                    ARM_NAMES[key],
                    cap,
                    queue["reviewed"],
                    queue["confirmed_illicit"],
                    queue["confirmed_licit"],
                    queue["reviewed_unknown"],
                    _number(queue["unknown_share"]),
                    _number(queue["recall_known_illicit"]),
                    _number(queue["expected_random_hits"]),
                    _number(queue["hits_over_expected"]),
                )
                for key, row in results.items()
                for cap, queue in row["q_all"].items()
            ),
        ),
        "## Cutoff ties at K=50",
        _table(
            (
                "Arm",
                "Mean tied at cut",
                "Steps with at most five tied",
                "Steps with ties spanning cut",
            ),
            (
                (
                    ARM_NAMES[key],
                    _number(row["cutoff_ties"]["mean"]),
                    f"{row['cutoff_ties']['small_ties']}/{row['cutoff_ties']['eligible_steps']}",
                    row["q_all"][50]["steps_with_ties_spanning_cut"],
                )
                for key, row in results.items()
            ),
        ),
        "## Descriptive split at step 43",
        _table(
            (
                "Arm",
                "40–42 mean step AP",
                "40–42 pooled AP",
                "43–49 mean step AP",
                "43–49 pooled AP",
            ),
            (
                (
                    ARM_NAMES[key],
                    *(
                        _number(row["split_at_43"][field])
                        for field in (
                            "before_43_mean_step_ap",
                            "before_43_pooled_ap",
                            "from_43_mean_step_ap",
                            "from_43_pooled_ap",
                        )
                    ),
                )
                for key, row in results.items()
            ),
        ),
        "## Operating points at the frozen F4 thresholds",
        _table(
            ("Arm", "Threshold", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"),
            (
                (
                    ARM_NAMES[key],
                    repr(row["threshold"]),
                    *(
                        _number(row["operating_point"][field])
                        for field in (
                            "tp",
                            "fp",
                            "fn",
                            "tn",
                            "precision",
                            "recall",
                            "f1",
                        )
                    ),
                )
                for key, row in results.items()
            ),
        ),
    ]


def _q_lab_step(row: dict, step: int, cap: int) -> str:
    queue = row["q_lab"][cap]["by_step"][step]
    return f"{_number(queue['precision'])} / {_number(queue['recall'])}"


def _queue_step(row: dict, step: int) -> tuple:
    q = row["q_all"][50]["by_step"][step]
    return (
        q["reviewed"],
        q["confirmed_illicit"],
        q["confirmed_licit"],
        q["reviewed_unknown"],
        _number(q["unknown_share"]),
        _number(q["recall_known_illicit"]),
        _number(q["expected_random_hits"]),
        _number(q["hits_over_expected"]),
        q["strictly_above"],
        q["tied_at_cut"],
        q["tied_selected"],
        "Yes" if q["spans_cut"] else "No",
    )


def _step_tables(results: dict) -> list[str]:
    support = results["s1_legacy"]["q_all"][50]["by_step"]
    return [
        "## Per-step support",
        "The eligible rows and known outcomes are shared across all four arms. Illicit prevalence uses known labels as the denominator.",
        _table(
            (
                "Step",
                "Eligible",
                "Labelled",
                "Illicit",
                "Licit",
                "Unlabelled",
                "Illicit prevalence",
            ),
            (
                (
                    step,
                    row["eligible"],
                    row["labelled"],
                    row["known_illicit"],
                    row["labelled"] - row["known_illicit"],
                    row["eligible"] - row["labelled"],
                    _number(
                        row["known_illicit"] / row["labelled"]
                        if row["labelled"]
                        else None
                    ),
                )
                for step, row in support.items()
            ),
        ),
        "## Per-step AP and Q-lab precision / recall",
        _table(
            ("Step", "Arm", "AP", "P / R at 10", "P / R at 25", "P / R at 50"),
            (
                (
                    step,
                    ARM_NAMES[key],
                    _number(row["per_step_ap"][step]),
                    *(_q_lab_step(row, step, cap) for cap in LABELLED_CAPACITIES),
                )
                for step in support
                for key, row in results.items()
            ),
        ),
        "## Per-step Q-all outcomes and cutoff ties at K=50",
        _table(
            (
                "Step",
                "Arm",
                "Reviewed",
                "Illicit",
                "Licit",
                "Unlabelled",
                "Unlabelled share",
                "Known-illicit recall",
                "Random hits expected",
                "Observed / expected",
                "Strictly above",
                "Tied at cut",
                "Tied and reviewed",
                "Spans cut",
            ),
            (
                (step, ARM_NAMES[key], *_queue_step(row, step))
                for step in support
                for key, row in results.items()
            ),
        ),
    ]


def render_evaluation(report: dict) -> str:
    """Render only aggregate, pre-registered results for publication."""
    results = report["arms"]
    attempt = report["access"]
    blocks = [
        "# Phase 5 final evaluation",
        f"One authorised evaluation of steps {TEST_STEPS[0]}–{TEST_STEPS[1]} under the [approved pre-registration](preregistration.md). "
        f"Access started {attempt['started_at']} on executable revision `{attempt['git_commit']}`. "
        f"The source release SHA-256 is `{attempt['source_snapshot_sha256']}`. "
        "All S2 results are conditional on A1. The deployed S1 model remains the legacy XGBoost.",
        f"Frozen shortlist SHA-256 `{attempt['freeze_sha256']}`. Approval revision `{attempt['approval_commit']}`.",
        "## Frozen arms",
        _table(
            ("Arm", "Frozen identity", "Model SHA-256"),
            (
                (ARM_NAMES[key], meta["name"], meta["model_sha256"])
                for key, meta in attempt["arms"].items()
            ),
        ),
        "The LightGBM model digest binds the five member digests in seed order.",
        "## Paired contrasts for mean step AP",
        "Each interval uses the same 1,000 paired whole-step bootstrap draws with seed 42. The two primary intervals are nominal 95% intervals without multiplicity adjustment. The third interval is descriptive.",
        _table(
            (
                "Role",
                "Contrast",
                "AP difference",
                "95% CI",
                "Paired steps",
                "Valid draws",
                "Conclusion",
            ),
            (
                (
                    row["role"],
                    row["label"],
                    _number(row["estimate"]),
                    _interval(row),
                    row["steps"],
                    row["defined_draws"],
                    row["conclusion"],
                )
                for row in report["contrasts"]
            ),
        ),
        *_summary_tables(results),
        *_step_tables(results),
        "## Interpretation and limitations",
        "- Conclusions apply to the ten evaluated steps. The limited known-illicit support in steps 45 and 46 makes AP sensitive to individual ranks. Both steps enter the registered analysis. An interval that contains zero leaves the difference inconclusive for this period.",
        "- The RF contrast compares modelling configurations at the same S2 feature contract. The legacy contrast compares the full scenarios. The descriptive S1 refit contrast matches the training horizon while feature availability and model families still differ.",
        "- A1 remains unverified. In development the matched two-step embargo changed the S2 LightGBM AP from 0.935226 to 0.904803. This sensitivity and the prior exposure recorded in the [holdout ledger](test-ledger.md) limit generalisation.",
        "- Unknown labels stay unlabelled. Q-all measures recovery of known illicit cases. F4 thresholds provide secondary operating points and the review queue uses top 50 per step.",
    ]
    return "\n\n".join(blocks) + "\n"
