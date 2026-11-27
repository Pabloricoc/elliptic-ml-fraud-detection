"""Aggregate descriptive statistics for a validated Elliptic dataset."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from .folds import DEV_STEPS
from .ingestion import EllipticDataset
from .schema import (
    CLASS_NAMES,
    CLASS_SCHEMA,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    TIME_STEPS,
    DatasetError,
)


@dataclass
class Summary:
    """Aggregate results that can be written directly to a local JSON report."""

    shapes: dict[str, tuple[int, int]]
    class_counts: dict[str, int]
    unknown_fraction: float
    known_class_shares: dict[str, float | None]
    graph: dict[str, int]
    constant_features: list[str]
    warnings: list[str]


def temporal_counts(transactions: pd.DataFrame) -> pd.DataFrame:
    """Count all source labels per release time step and retain undefined ratios."""
    table = (
        pd.crosstab(transactions["time_step"], transactions["class"])
        .reindex(index=TIME_STEPS, columns=list(CLASS_NAMES), fill_value=0)
        .rename(columns=CLASS_NAMES)
        .rename_axis(index="time_step", columns=None)
    )
    table["transactions"] = table.sum(axis=1)
    table["labelled"] = table["illicit"] + table["licit"]
    table["label_coverage"] = table["labelled"].div(
        table["transactions"].replace(0, float("nan"))
    )
    table["illicit_share_labelled"] = table["illicit"].div(
        table["labelled"].replace(0, float("nan"))
    )
    return table


def _graph_counts(edges: pd.DataFrame, nodes: int) -> dict[str, int]:
    outgoing = edges["source_id"].value_counts()
    incoming = edges["target_id"].value_counts()
    connected = len(outgoing.index.union(incoming.index, sort=False))
    return {
        "nodes": nodes,
        "directed_edges": len(edges),
        "isolated_nodes": nodes - connected,
        "duplicate_edge_rows": int(edges.duplicated().sum()),
        "self_loops": int(edges["source_id"].eq(edges["target_id"]).sum()),
        "max_in_degree": int(incoming.max()) if len(incoming) else 0,
        "max_out_degree": int(outgoing.max()) if len(outgoing) else 0,
    }


def summarize(dataset: EllipticDataset) -> tuple[Summary, pd.DataFrame]:
    """Return an aggregate overview and its chronological label counts."""
    tx, edges = dataset.transactions, dataset.edges
    periods = temporal_counts(tx)
    counts = {name: int(periods[name].sum()) for name in CLASS_NAMES.values()}
    known = counts["licit"] + counts["illicit"]
    graph = _graph_counts(edges, len(tx))
    constant = [name for name in FEATURE_COLUMNS if tx[name].min() == tx[name].max()]
    warnings = []
    if len(tx) != 203_769 or len(edges) != 234_355:
        warnings.append("Table sizes differ from the published full release.")
    if (periods["transactions"] == 0).any():
        warnings.append("Some of the 49 release time steps have no transactions.")
    if ((periods["transactions"] > 0) & (periods["labelled"] == 0)).any():
        warnings.append("Some observed time steps have no known labels.")
    if counts["licit"] == 0 or counts["illicit"] == 0:
        warnings.append("At least one known class is absent.")
    if constant:
        warnings.append(f"{len(constant)} numeric feature columns are constant.")
    if graph["duplicate_edge_rows"] or graph["self_loops"]:
        warnings.append("The directed edge list contains repeated edges or self-loops.")
    summary = Summary(
        shapes={
            "features": (len(tx), len(FEATURE_SCHEMA)),
            "classes": (len(tx), len(CLASS_SCHEMA)),
            "edges": edges.shape,
            "joined_transactions": tx.shape,
        },
        class_counts=counts,
        unknown_fraction=counts["unknown"] / len(tx),
        known_class_shares={
            name: counts[name] / known if known else None
            for name in ("illicit", "licit")
        },
        graph=graph,
        constant_features=constant,
        warnings=warnings,
    )
    return summary, periods


def feature_statistics(transactions: pd.DataFrame) -> pd.DataFrame:
    """Describe each anonymous feature separately to bound working memory."""
    stats = pd.DataFrame(
        {name: transactions[name].describe() for name in FEATURE_COLUMNS}
    ).T.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
    stats.index.name = "feature"
    stats["count"] = stats["count"].astype(int)
    return stats


def a1_probe(dataset: EllipticDataset, train: tuple[int, int]) -> dict:
    """Rank anonymous columns by absolute partial Spearman on fold training nodes."""
    first, last = train
    if not DEV_STEPS[0] <= first <= last <= DEV_STEPS[1]:
        raise DatasetError("The A1 probe requires a development training window.")
    rows = dataset.transactions
    if not rows["time_step"].between(*DEV_STEPS).all():
        raise DatasetError("The A1 probe accepts development-only inputs.")
    rows = rows.loc[rows["time_step"].between(first, last)].set_index("tx_id")
    if rows.empty:
        raise DatasetError("The A1 probe training window is empty.")
    edges = dataset.edges
    edges = edges.loc[edges["target_id"].isin(rows.index)].drop_duplicates()
    if not edges["source_id"].isin(rows.index).all():
        raise DatasetError("The A1 probe received a funding edge outside training.")

    incoming = edges["target_id"].value_counts().reindex(rows.index, fill_value=0)
    outgoing = edges["source_id"].value_counts().reindex(rows.index, fill_value=0)
    control = incoming.rank().to_numpy()
    control -= control.mean()
    variance = float(np.sum(control * control))

    def residual_rank(values: pd.Series) -> np.ndarray:
        ranked = values.rank().to_numpy()
        ranked -= ranked.mean()
        if variance:
            ranked -= float(np.sum(ranked * control)) / variance * control
        return ranked

    out_residual = residual_rank(outgoing)
    out_norm = np.sqrt(np.sum(out_residual**2))
    coefficients = {}
    for column in FEATURE_COLUMNS:
        residual = residual_rank(rows[column])
        denominator = out_norm * np.sqrt(np.sum(residual**2))
        coefficients[column] = (
            float(np.clip(np.sum(residual * out_residual) / denominator, -1, 1))
            if denominator > np.finfo(float).eps
            else None
        )
    ordered = sorted(
        (column for column, value in coefficients.items() if value is not None),
        key=lambda column: (-abs(coefficients[column]), column),
    )
    return {
        "train": list(train),
        "transactions": len(rows),
        "partial_spearman": coefficients,
        "flagged": ordered[: ceil(0.05 * len(ordered))],
    }
