"""Aggregate descriptive statistics for a validated Elliptic dataset."""

from dataclasses import dataclass

import pandas as pd

from .ingestion import EllipticDataset
from .schema import (
    CLASS_NAMES,
    CLASS_SCHEMA,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    TIME_STEPS,
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
