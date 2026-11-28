"""Label-aware supervised metrics and a fixed-capacity temporal review queue."""

import hashlib
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .model_features import Window


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def average_precision(target: pd.Series, scores: pd.Series) -> float | None:
    """Return illicit-class AP on aligned known outcomes, undefined without positives."""
    if not target.sum():
        return None
    return float(average_precision_score(target, scores.loc[target.index]))


def pooled_ap(target: pd.Series, scores: pd.Series) -> float | None:
    """Compute AP on known labels pooled across time steps."""
    return average_precision(target, scores)


def per_step_ap(window: Window, scores: pd.Series) -> dict[int, float | None]:
    """Compute AP for each requested step, including undefined and empty steps."""
    known = window.known_target()
    grouped = dict(tuple(known.groupby(window.steps.loc[known.index])))
    return {
        step: average_precision(grouped[step], scores) if step in grouped else None
        for step in range(window.first, window.last + 1)
    }


def roc_auc(target: pd.Series, scores: pd.Series) -> float | None:
    """Compute ROC-AUC on the known labels when both classes are present."""
    if target.nunique() != 2:
        return None
    return float(roc_auc_score(target, scores.loc[target.index]))


def brier_score(target: pd.Series, scores: pd.Series) -> float | None:
    """Compute the mean squared probability error on known labels."""
    if target.empty:
        return None
    return float(brier_score_loss(target, scores.loc[target.index]))


def precision_recall_at_k_labelled(
    window: Window, scores: pd.Series, k: int, seed: int
) -> dict[int, dict[str, int | float | None]]:
    """Evaluate the top min(K, known labels) of each labelled-only ranking."""
    if k < 1:
        raise ValueError("Labelled review capacity must be positive.")
    known = window.known_target()
    ranked = rank_transactions(window.steps.loc[known.index], scores, seed)
    outcomes = ranked.assign(illicit=known.reindex(ranked.index))
    positives = outcomes.groupby("time_step")["illicit"].sum()
    reviewed = outcomes.loc[outcomes["review_rank"].le(k)]
    hits = reviewed.groupby("time_step")["illicit"].agg(["sum", "size"])
    hits = hits.reindex(range(window.first, window.last + 1), fill_value=0)
    return {
        int(step): {
            "reviewed_labelled": int(row.size),
            "confirmed_illicit": int(row.sum),
            "precision": _ratio(int(row.sum), int(row.size)),
            "recall": _ratio(int(row.sum), int(positives.get(step, 0))),
        }
        for step, row in zip(hits.index, hits.itertuples(index=False), strict=True)
    }


def queue_vs_random(by_step: pd.DataFrame) -> dict[str, float | None]:
    """Compare known illicit hits with sampling the same capacity uniformly per step."""
    expected = math.fsum(
        int(reviewed) * int(illicit) / int(eligible)
        for reviewed, illicit, eligible in by_step[
            ["reviewed", "known_illicit", "eligible"]
        ].itertuples(index=False, name=None)
        if eligible
    )
    return {
        "expected_random_hits": expected,
        "hits_over_expected": _ratio(int(by_step["confirmed_illicit"].sum()), expected),
    }


def tie_mass_at_cut(
    steps: pd.Series, scores: pd.Series, first: int, last: int, k: int
) -> dict[int, dict[str, int | bool]]:
    """Count scores above and tied at the cut, including blocks spanning the boundary."""
    if k < 1:
        raise ValueError("Review capacity must be positive.")
    result = {}
    for step in range(first, last + 1):
        values = scores.loc[steps[steps.eq(step)].index].to_numpy()
        slots = min(k, len(values))
        above = tied = 0
        if slots:
            cut_index = len(values) - slots
            cut = np.partition(values, cut_index)[cut_index]
            above = int(np.count_nonzero(values > cut))
            tied = int(np.count_nonzero(values == cut))
        result[step] = {
            "strictly_above": above,
            "tied_at_cut": tied,
            "tied_selected": slots - above,
            "spans_cut": above + tied > slots,
        }
    return result


def block_bootstrap_ci(
    values: Mapping[int, float | None],
    *,
    draws: int = 1000,
    seed: int = 42,
    indices: np.ndarray | None = None,
) -> dict[str, int | float | None]:
    """95% percentile interval for a step-macro mean by resampling whole steps."""
    if draws < 1:
        raise ValueError("Bootstrap draw count must be positive.")
    blocks = np.array([values[step] for step in sorted(values)], dtype=float)
    defined = blocks[~np.isnan(blocks)]
    if not len(defined):
        return {"estimate": None, "lower": None, "upper": None, "defined_draws": 0}
    sampled = (
        np.random.default_rng(seed).choice(blocks, size=(draws, len(blocks)))
        if indices is None
        else blocks[indices]
    )
    counts = np.count_nonzero(~np.isnan(sampled), axis=1)
    estimates = np.nansum(sampled, axis=1)[counts > 0] / counts[counts > 0]
    lower, upper = (
        (float(value) for value in np.quantile(estimates, [0.025, 0.975]))
        if len(estimates)
        else (None, None)
    )
    return {
        "estimate": float(defined.mean()),
        "lower": lower,
        "upper": upper,
        "defined_draws": len(estimates),
    }


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


def rank_transactions(steps: pd.Series, scores: pd.Series, seed: int) -> pd.DataFrame:
    """Rank descending within each step, using the original deterministic tie rule."""
    ranked = pd.DataFrame({"time_step": steps, "risk_score": scores.loc[steps.index]})
    ranked["tie"] = [
        hashlib.sha256(f"{seed}|{tx_id}".encode("ascii")).digest()
        for tx_id in ranked.index
    ]
    ranked = ranked.sort_values(
        ["time_step", "risk_score", "tie"], ascending=[True, False, True]
    )
    ranked["review_rank"] = ranked.groupby("time_step").cumcount() + 1
    return ranked.drop(columns="tie")


def review_queue(
    window: Window, scores: pd.Series, budget: int, seed: int
) -> tuple[dict[str, int | float | None], pd.DataFrame]:
    """Rank all transactions before looking up outcomes. Export step aggregates only."""
    if budget < 1:
        raise ValueError("Review budget must be positive.")
    ranked = rank_transactions(window.steps, scores, seed)
    rows = []
    for step in range(window.first, window.last + 1):
        candidates = ranked.index[ranked["time_step"].eq(step)]
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
