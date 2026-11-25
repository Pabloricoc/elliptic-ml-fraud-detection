"""Label-aware supervised metrics and a fixed-capacity temporal review queue."""

import hashlib

import pandas as pd
from sklearn.metrics import average_precision_score

from .model_features import Window


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def average_precision(target: pd.Series, scores: pd.Series) -> float | None:
    """Return illicit-class AP on aligned known outcomes, undefined without positives."""
    if not target.sum():
        return None
    return float(average_precision_score(target, scores.loc[target.index]))


def validation_threshold(target: pd.Series, scores: pd.Series) -> float:
    """Maximise validation F1, taking the highest threshold on a tie."""
    positives = int(target.sum())
    if not positives:
        raise ValueError("Threshold selection needs a labelled positive.")
    # Descending order makes the first maximum the highest threshold. All
    # observations with identical scores must enter the confusion matrix together.
    groups = target.groupby(scores.loc[target.index]).agg(["sum", "size"])
    alerts = groups.sort_index(ascending=False).cumsum()
    f1 = (2 * alerts["sum"]) / (positives + alerts["size"])
    return float(f1.idxmax())


def at_threshold(
    target: pd.Series, scores: pd.Series, threshold: float
) -> dict[str, int | float | None]:
    """Count labelled outcomes at a previously fixed operating point."""
    alerts = scores.loc[target.index].ge(threshold)
    positive = target.eq(1)
    tp = int((alerts & positive).sum())
    fp = int((alerts & ~positive).sum())
    fn = int((~alerts & positive).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": len(target) - tp - fp - fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def review_queue(
    window: Window, scores: pd.Series, budget: int, seed: int
) -> tuple[dict[str, int | float | None], pd.DataFrame]:
    """Rank all transactions before looking up outcomes. Export step aggregates only."""
    if budget < 1:
        raise ValueError("Review budget must be positive.")
    ranked = pd.DataFrame(
        {"step": window.steps, "score": scores.loc[window.steps.index]}
    )
    ranked["tie"] = [
        hashlib.sha256(f"{seed}|{tx_id}".encode("ascii")).digest()
        for tx_id in ranked.index
    ]
    ranked = ranked.sort_values(["step", "score", "tie"], ascending=[True, False, True])
    rows = []
    for step in range(window.first, window.last + 1):
        candidates = ranked.index[ranked["step"].eq(step)]
        outcomes = window.labels.loc[candidates]
        reviewed = outcomes.iloc[:budget]
        rows.append(
            {
                "time_step": step,
                "eligible": len(outcomes),
                "labelled": int(outcomes.ne("unknown").sum()),
                "known_illicit": int(outcomes.eq("1").sum()),
                "reviewed": len(reviewed),
                "confirmed_illicit": int(reviewed.eq("1").sum()),
                "confirmed_licit": int(reviewed.eq("2").sum()),
                "reviewed_unknown": int(reviewed.eq("unknown").sum()),
            }
        )
    table = pd.DataFrame(rows).set_index("time_step")
    totals = {key: int(value) for key, value in table.sum().items()}
    table["recall_known_illicit"] = table["confirmed_illicit"].div(
        table["known_illicit"].where(table["known_illicit"] > 0)
    )
    return {
        **totals,
        "budget_per_step": budget,
        "recall_known_illicit": _ratio(
            totals["confirmed_illicit"], totals["known_illicit"]
        ),
    }, table
