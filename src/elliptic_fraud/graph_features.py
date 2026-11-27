"""Graph features computed independently within a single time step."""

from graphlib import CycleError, TopologicalSorter

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, DatasetError

LOCAL_COLUMNS = FEATURE_COLUMNS[:93]
AGGREGATED_COLUMNS = FEATURE_COLUMNS[93:]
ANCESTRY_COLUMNS = (
    "observed_ancestor_count",
    "observed_two_hop_ancestor_count",
    "observed_ancestor_depth",
    "mean_ancestor_indegree",
    "max_ancestor_indegree",
    "std_ancestor_indegree",
    "observed_ancestor_root_count",
)
BATCH_GRAPH_COLUMNS = (
    "observed_out_degree",
    "observed_descendant_count",
    "weak_component_size",
    "within_step_pagerank",
)
PARENT_COLUMNS = tuple(
    f"{stat}_{feature}"
    for stat in ("parent_mean", "parent_max", "parent_std", "grandparent_mean")
    for feature in LOCAL_COLUMNS
)


def _parent_graph(ids: pd.Index, edges: pd.DataFrame) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = {tx_id: set() for tx_id in ids}
    for source, target in edges[["source_id", "target_id"]].itertuples(
        index=False, name=None
    ):
        if source not in parents or target not in parents:
            raise DatasetError(
                "A funding edge refers to a transaction outside the step."
            )
        parents[target].add(source)
    return parents


def _topological_order(parents: dict[str, set[str]]) -> list[str]:
    try:
        return list(TopologicalSorter(parents).static_order())
    except CycleError:
        raise DatasetError(
            "A directed cycle prevents funding-graph features."
        ) from None


def _grandparents(parents: dict[str, set[str]], tx_id: str) -> set[str]:
    return {ancestor for parent in parents[tx_id] for ancestor in parents[parent]}


def extended_ancestry(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize distinct funding ancestors and their observed in-degrees."""
    parents = _parent_graph(nodes.index, edges)
    ancestors: dict[str, set[str]] = {}
    depths: dict[str, int] = {}
    values = {}
    for tx_id in _topological_order(parents):
        incoming = parents[tx_id]
        upstream = incoming.union(*(ancestors[parent] for parent in incoming))
        ancestors[tx_id] = upstream
        depth = 1 + max((depths[parent] for parent in incoming), default=-1)
        depths[tx_id] = depth
        degrees = np.fromiter(
            (len(parents[ancestor]) for ancestor in upstream), dtype="int64"
        )
        degrees.sort()
        values[tx_id] = (
            len(upstream),
            len(_grandparents(parents, tx_id)),
            depth,
            degrees.mean() if upstream else 0,
            degrees.max() if upstream else 0,
            degrees.std() if upstream else 0,
            int(np.count_nonzero(degrees == 0)),
        )
    return pd.DataFrame.from_dict(
        values, orient="index", columns=ANCESTRY_COLUMNS, dtype="float64"
    ).reindex(nodes.index)


def _component_sizes(
    parents: dict[str, set[str]], children: dict[str, set[str]]
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for root in parents:
        if root in sizes:
            continue
        seen = {root}
        pending = [root]
        while pending:
            tx_id = pending.pop()
            neighbours = (parents[tx_id] | children[tx_id]) - seen
            seen.update(neighbours)
            pending.extend(neighbours)
        sizes.update(dict.fromkeys(seen, len(seen)))
    return sizes


def _pagerank(ids: pd.Index, edges: pd.DataFrame) -> np.ndarray:
    edge_pairs = edges.drop_duplicates()
    source = ids.get_indexer(edge_pairs["source_id"])
    target = ids.get_indexer(edge_pairs["target_id"])
    size, damping = len(ids), 0.85
    degree = np.bincount(source, minlength=size)
    rank = np.full(size, 1 / size)
    for _ in range(200):
        incoming = np.bincount(
            target, weights=rank[source] / degree[source], minlength=size
        )
        update = (
            damping * incoming
            + (1 - damping + damping * rank[degree == 0].sum()) / size
        )
        if np.abs(update - rank).sum() <= 1e-12:
            return update
        rank = update
    raise DatasetError("Within-step PageRank did not converge.")


def batch_structure(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute outgoing and connectivity features at the end of a step."""
    parents = _parent_graph(nodes.index, edges)
    children: dict[str, set[str]] = {tx_id: set() for tx_id in nodes.index}
    for tx_id, incoming in parents.items():
        for parent in incoming:
            children[parent].add(tx_id)
    descendants: dict[str, set[str]] = {}
    for tx_id in reversed(_topological_order(parents)):
        outgoing = children[tx_id]
        descendants[tx_id] = outgoing.union(*(descendants[child] for child in outgoing))
    sizes = _component_sizes(parents, children)
    return pd.DataFrame(
        {
            BATCH_GRAPH_COLUMNS[0]: [len(children[tx_id]) for tx_id in nodes.index],
            BATCH_GRAPH_COLUMNS[1]: [len(descendants[tx_id]) for tx_id in nodes.index],
            BATCH_GRAPH_COLUMNS[2]: [sizes[tx_id] for tx_id in nodes.index],
            BATCH_GRAPH_COLUMNS[3]: _pagerank(nodes.index, edges),
        },
        index=nodes.index,
        dtype="float64",
    )


def parent_aggregates(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize each local attribute over distinct parents and grandparents."""
    parents = _parent_graph(nodes.index, edges)
    _topological_order(parents)
    local = nodes.loc[:, list(LOCAL_COLUMNS)].to_numpy(dtype="float64")
    width = len(LOCAL_COLUMNS)
    rows = np.zeros((len(nodes), len(PARENT_COLUMNS)))
    ids = {tx_id: i for i, tx_id in enumerate(nodes.index)}
    for i, tx_id in enumerate(nodes.index):
        if parents[tx_id]:
            direct = local[sorted(ids[parent] for parent in parents[tx_id])]
            rows[i, :width] = direct.mean(axis=0)
            rows[i, width : 2 * width] = direct.max(axis=0)
            rows[i, 2 * width : 3 * width] = direct.std(axis=0)
        upstream = _grandparents(parents, tx_id)
        if upstream:
            second = local[sorted(ids[ancestor] for ancestor in upstream)]
            rows[i, 3 * width :] = second.mean(axis=0)
    return pd.DataFrame(rows, index=nodes.index, columns=PARENT_COLUMNS)
