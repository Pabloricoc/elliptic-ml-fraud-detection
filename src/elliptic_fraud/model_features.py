"""Construct confirmation-time predictors from the observed incoming payment graph."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter

import pandas as pd

from .graph_features import (
    AGGREGATED_COLUMNS,
    ANCESTRY_COLUMNS,
    BATCH_GRAPH_COLUMNS,
    LOCAL_COLUMNS,
    PARENT_COLUMNS,
    batch_structure,
    extended_ancestry,
    parent_aggregates,
)
from .ingestion import EllipticDataset
from .schema import TIME_STEPS, DatasetError

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


def scoring_features(
    dataset: EllipticDataset, first: int, last: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Select complete steps and build ancestry with no dependency on outcomes."""
    tx = dataset.transactions
    selected = tx.loc[
        tx["time_step"].between(first, last), ["tx_id", "time_step"]
    ].set_index("tx_id")["time_step"]
    if selected.empty:
        raise DatasetError(f"No eligible transactions in time steps {first}–{last}.")
    edges = dataset.edges.loc[dataset.edges["target_id"].isin(selected.index)]
    return ancestry_features(selected.index, edges), selected


def build_window(dataset: EllipticDataset, first: int, last: int) -> Window:
    """Attach outcomes after computing the predictors for a complete window."""
    features, steps = scoring_features(dataset, first, last)
    labels = dataset.transactions[["tx_id", "class"]].set_index("tx_id")["class"]
    return Window(features, labels.loc[features.index], steps, first, last)


@dataclass(frozen=True)
class FeatureSet:
    """A versioned predictor contract and its per-step construction rule."""

    name: str
    version: int
    scenario: str
    columns: tuple[str, ...]
    builder: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    availability: str
    required_columns: tuple[str, ...] = ()

    def manifest(self) -> dict:
        """Provide the configuration fields stored in a version 2 model manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "scenario": self.scenario,
            "columns": list(self.columns),
        }


def _legacy_ancestry(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    return ancestry_features(nodes.index, edges)


def _select_columns(
    columns: tuple[str, ...],
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    def select(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
        return nodes.loc[:, list(columns)].astype("float64")

    return select


_ANCESTRY = (
    FeatureSet(
        "T0",
        1,
        "S1",
        tuple(FEATURE_AUDIT),
        _legacy_ancestry,
        "Incoming funding edges and their ancestry at confirmation.",
    ),
    FeatureSet(
        "T1",
        1,
        "S1",
        ANCESTRY_COLUMNS,
        extended_ancestry,
        "Distinct ancestors and their incoming funding edges at confirmation.",
    ),
)
_BATCH = (
    FeatureSet(
        "T1b",
        1,
        "S2",
        BATCH_GRAPH_COLUMNS,
        batch_structure,
        "Complete observed funding graph of the step at its close.",
    ),
    FeatureSet(
        "T2",
        1,
        "S2",
        LOCAL_COLUMNS,
        _select_columns(LOCAL_COLUMNS),
        "Anonymous local columns available at step close conditional on A1.",
        LOCAL_COLUMNS,
    ),
    FeatureSet(
        "T2agg",
        1,
        "S2",
        PARENT_COLUMNS,
        parent_aggregates,
        "Incoming ancestors' local columns at step close conditional on A1.",
        LOCAL_COLUMNS,
    ),
    FeatureSet(
        "T3",
        1,
        "S2",
        AGGREGATED_COLUMNS,
        _select_columns(AGGREGATED_COLUMNS),
        "Anonymous neighbour aggregates at step close conditional on A1.",
        AGGREGATED_COLUMNS,
    ),
)
FEATURE_SETS = {
    (tier.scenario, tier.name): tier
    for tier in (
        *_ANCESTRY,
        *(replace(tier, scenario="S2") for tier in _ANCESTRY),
        *_BATCH,
    )
}


def _ordered_union(columns: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(column for group in columns for column in group))


def get_feature_set(name: str, scenario: str, version: int = 1) -> FeatureSet:
    """Resolve a registered tier or ordered union of tiers for one scenario."""
    if type(version) is not int or version != 1:
        raise DatasetError("Unsupported feature-set version.")
    names = name.split("+")
    if len(set(names)) != len(names):
        raise DatasetError("A composite feature set must list each tier once.")
    try:
        tiers = [FEATURE_SETS[scenario, part] for part in names]
    except KeyError:
        raise DatasetError(
            "Unsupported feature set for the requested scenario."
        ) from None
    if len(tiers) == 1:
        return tiers[0]
    columns = _ordered_union(tier.columns for tier in tiers)

    def build(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([tier.builder(nodes, edges) for tier in tiers], axis=1)
        return combined.loc[:, ~combined.columns.duplicated()]

    return FeatureSet(
        name,
        version,
        scenario,
        columns,
        build,
        " ".join(tier.availability for tier in tiers),
        _ordered_union(tier.required_columns for tier in tiers),
    )


def build_feature_set(
    dataset: EllipticDataset, feature_set: FeatureSet, first: int, last: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Build complete steps independently and restore input transaction order."""
    if first not in TIME_STEPS or last not in TIME_STEPS or first > last:
        raise DatasetError("Choose an inclusive feature window from 1 to 49.")
    columns = ["tx_id", "time_step", *feature_set.required_columns]
    transactions = dataset.transactions
    if any(column not in transactions for column in columns):
        raise DatasetError(
            "The selected feature set requires additional input columns."
        )
    selected = transactions.loc[transactions["time_step"].between(first, last), columns]
    steps = selected.set_index("tx_id")["time_step"]
    if steps.empty or not steps.index.is_unique:
        raise DatasetError(
            "The feature window requires nonempty, unique transaction IDs."
        )
    relevant = dataset.edges.loc[dataset.edges["target_id"].isin(steps.index)]
    target_steps = relevant["target_id"].map(steps)
    if relevant["source_id"].map(steps).ne(target_steps).any():
        raise DatasetError("A funding edge crosses time steps or has a missing source.")
    # Strip identifiers, steps and labels before invoking the predictor builders.
    attributes = selected.set_index("tx_id").drop(columns="time_step")
    edges_by_step = dict(tuple(relevant.groupby(target_steps)))
    parts = [
        feature_set.builder(
            attributes.loc[step_ids.index], edges_by_step.get(step, relevant.iloc[:0])
        )
        for step, step_ids in steps.groupby(steps)
    ]
    features = pd.concat(parts).reindex(steps.index)
    return features, steps
