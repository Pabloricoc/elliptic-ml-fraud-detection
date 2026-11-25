"""Construct confirmation-time predictors from the observed incoming payment graph."""

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

import pandas as pd

from .ingestion import EllipticDataset
from .schema import DatasetError

FEATURE_AUDIT = {
    "observed_parent_count": "Number of distinct observed transactions funding this transaction.",
    "mean_parent_indegree": "Mean number of observed parents of those funding transactions.",
    "max_parent_indegree": "Maximum number of observed parents of those funding transactions.",
    "observed_ancestor_depth": "Longest observed directed funding path ending at this transaction.",
}
WINDOWS = {"train": (1, 29), "validation": (30, 39), "test": (40, 49)}


@dataclass
class Window:
    """Predictors for all eligible transactions and separately indexed audit fields."""

    features: pd.DataFrame
    labels: pd.Series
    steps: pd.Series
    first: int
    last: int

    def known_target(self) -> pd.Series:
        labelled = self.labels[self.labels.isin(["1", "2"])]
        return labelled.eq("1").astype("int8").rename("illicit")

    def support(self) -> dict[str, int | float]:
        target = self.known_target()
        return {
            "first_step": self.first,
            "last_step": self.last,
            "transactions": len(self.labels),
            "labelled": len(target),
            "illicit": int(target.sum()),
            "licit": int(target.eq(0).sum()),
            "unknown": len(self.labels) - len(target),
            "label_coverage": len(target) / len(self.labels),
        }


def ancestry_features(ids: pd.Index, edges: pd.DataFrame) -> pd.DataFrame:
    """Build predictors using parents and their ancestors, including unlabelled nodes."""
    parents: dict[str, set[str]] = {tx_id: set() for tx_id in ids}
    for source, target in edges[["source_id", "target_id"]].itertuples(
        index=False, name=None
    ):
        if source not in parents or target not in parents:
            raise DatasetError(
                "A funding edge refers to a transaction outside the window."
            )
        parents[target].add(source)

    depths: dict[str, int] = {}
    rows = {}
    try:
        for tx_id in TopologicalSorter(parents).static_order():
            incoming = parents[tx_id]
            parent_degrees = [len(parents[parent]) for parent in incoming]
            depth = 1 + max((depths[parent] for parent in incoming), default=-1)
            depths[tx_id] = depth
            rows[tx_id] = (
                len(incoming),
                sum(parent_degrees) / len(incoming) if incoming else 0.0,
                max(parent_degrees, default=0),
                depth,
            )
    except CycleError:
        raise DatasetError(
            "A directed cycle prevents confirmation-time ancestry features."
        ) from None
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=list(FEATURE_AUDIT), dtype="float64"
    ).reindex(ids)


def build_window(dataset: EllipticDataset, first: int, last: int) -> Window:
    """Select whole steps before feature construction. Preserve unknowns in the graph."""
    tx = dataset.transactions
    selected = tx.loc[
        tx["time_step"].between(first, last), ["tx_id", "class", "time_step"]
    ].set_index("tx_id")
    if selected.empty:
        raise DatasetError(f"No eligible transactions in time steps {first}–{last}.")
    edges = dataset.edges.loc[dataset.edges["target_id"].isin(selected.index)]
    features = ancestry_features(selected.index, edges)
    return Window(features, selected["class"], selected["time_step"], first, last)
