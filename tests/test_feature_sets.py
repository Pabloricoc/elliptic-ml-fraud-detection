from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elliptic_fraud import feature_cache
from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.model_features import (
    FEATURE_AUDIT,
    build_feature_set,
    get_feature_set,
    scoring_features,
)
from elliptic_fraud.runs import file_digest, read_json
from elliptic_fraud.schema import FEATURE_COLUMNS, DatasetError


def _transactions(ids: list[str], step: int) -> pd.DataFrame:
    data = np.array([float(tx_id) for tx_id in ids])[:, None] + np.arange(165)
    return pd.DataFrame(data, columns=FEATURE_COLUMNS).assign(
        tx_id=ids, time_step=step, **{"class": "unknown"}
    )


@pytest.fixture
def step_graph() -> EllipticDataset:
    nodes = _transactions(["5", "1", "2", "4", "3"], 1)
    edges = pd.DataFrame(
        [("1", "2"), ("1", "3"), ("2", "4"), ("3", "4"), ("1", "4"), ("2", "4")],
        columns=["source_id", "target_id"],
    )
    return EllipticDataset(nodes, edges)


def test_feature_contracts_preserve_t0_and_use_the_documented_columns(
    step_graph: EllipticDataset,
) -> None:
    legacy, legacy_steps = scoring_features(step_graph, 1, 1)
    t0, steps = build_feature_set(step_graph, get_feature_set("T0", "S1"), 1, 1)
    pd.testing.assert_frame_equal(t0, legacy, check_exact=True)
    pd.testing.assert_series_equal(steps, legacy_steps)
    assert list(t0.columns) == list(FEATURE_AUDIT)

    local = get_feature_set("T2", "S2")
    aggregated = get_feature_set("T3", "S2")
    assert local.columns == FEATURE_COLUMNS[:93] and local.columns[-1] == "feature_093"
    assert (
        aggregated.columns == FEATURE_COLUMNS[93:]
        and aggregated.columns[0] == "feature_094"
    )
    assert not {"time_step", "tx_id", "class"} & set(local.columns + aggregated.columns)
    combo = get_feature_set("T0+T1", "S1")
    actual, _ = build_feature_set(step_graph, combo, 1, 1)
    assert tuple(actual.columns) == combo.columns and actual.columns.is_unique
    pd.testing.assert_frame_equal(actual.loc[:, list(t0.columns)], t0, check_exact=True)
    for forbidden in ("T1b", "T2", "T2agg", "T3", "T1+T2"):
        with pytest.raises(DatasetError, match="requested scenario"):
            get_feature_set(forbidden, "S1")
    with pytest.raises(DatasetError, match="each tier once"):
        get_feature_set("T0+T0", "S1")


def test_graph_tiers_measure_distinct_ancestors_descendants_and_parents(
    step_graph: EllipticDataset,
) -> None:
    t1, _ = build_feature_set(step_graph, get_feature_set("T1", "S1"), 1, 1)
    assert list(t1.loc["4"]) == pytest.approx([3, 1, 2, 2 / 3, 1, np.sqrt(2) / 3, 1])
    assert not t1.loc["5"].any()
    batch, _ = build_feature_set(step_graph, get_feature_set("T1b", "S2"), 1, 1)
    assert list(batch.loc["1"].iloc[:3]) == [3, 3, 4]
    assert list(batch.loc["5"].iloc[:3]) == [0, 0, 1]
    assert batch["within_step_pagerank"].sum() == pytest.approx(1)
    assert (
        batch.loc["4", "within_step_pagerank"] > batch.loc["1", "within_step_pagerank"]
    )
    parent, _ = build_feature_set(step_graph, get_feature_set("T2agg", "S2"), 1, 1)
    assert list(
        parent.loc[
            "4",
            [
                "parent_mean_feature_001",
                "parent_max_feature_001",
                "parent_std_feature_001",
                "grandparent_mean_feature_001",
            ],
        ]
    ) == pytest.approx([2, 3, np.sqrt(2 / 3), 1])
    assert not parent.loc["5"].any()


def test_parent_presence_distinguishes_real_zero_values_from_absent_groups(
    step_graph: EllipticDataset,
) -> None:
    step_graph.transactions.loc[
        step_graph.transactions["tx_id"].eq("1"), "feature_001"
    ] = 0
    features, _ = build_feature_set(step_graph, get_feature_set("T2agg", "S2"), 1, 1)
    assert list(features.loc[["5", "2"], "parent_mean_feature_001"]) == [0, 0]
    assert list(features.loc[["5", "2"], "has_parents"]) == [0, 1]
    assert list(features.loc[["2", "4"], "grandparent_mean_feature_001"]) == [0, 0]
    assert list(features.loc[["2", "4"], "has_grandparents"]) == [0, 1]


@pytest.mark.parametrize(
    ("scenario", "name"),
    [
        ("S1", "T0"),
        ("S1", "T1"),
        ("S2", "T1b"),
        ("S2", "T2"),
        ("S2", "T2agg"),
        ("S2", "T3"),
        ("S2", "T1b+T2+T3+T2agg"),
    ],
)
def test_all_tiers_isolate_later_steps(
    scenario: str, name: str, step_graph: EllipticDataset
) -> None:
    feature_set = get_feature_set(name, scenario)
    before, _ = build_feature_set(step_graph, feature_set, 1, 1)
    added = _transactions(["8", "9"], 2)
    edges = pd.DataFrame([("8", "9")], columns=step_graph.edges.columns)
    combined = EllipticDataset(
        pd.concat([added, step_graph.transactions], ignore_index=True),
        pd.concat([edges, step_graph.edges], ignore_index=True),
    )
    after, steps = build_feature_set(combined, feature_set, 1, 2)
    pd.testing.assert_frame_equal(before, after.loc[steps.eq(1)], check_exact=True)
    assert list(after.index[:2]) == ["8", "9"]


@pytest.mark.parametrize("name", ["T0", "T1", "T0+T1"])
def test_s1_is_invariant_to_appended_descendants(
    name: str, step_graph: EllipticDataset
) -> None:
    feature_set = get_feature_set(name, "S1")
    before, _ = build_feature_set(step_graph, feature_set, 1, 1)
    added = _transactions(["6", "7"], 1)
    edges = pd.DataFrame([("4", "6"), ("6", "7")], columns=step_graph.edges.columns)
    combined = EllipticDataset(
        pd.concat([step_graph.transactions, added], ignore_index=True),
        pd.concat([step_graph.edges, edges], ignore_index=True),
    )
    after, _ = build_feature_set(combined, feature_set, 1, 1)
    pd.testing.assert_frame_equal(before, after.loc[before.index], check_exact=True)


def test_cache_reuses_the_registered_snapshot_and_excludes_holdout(
    step_graph: EllipticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = feature_cache.cached_development_features
    fingerprint = feature_cache.development_snapshot_digest
    snapshot = fingerprint(step_graph)
    t0, t1 = get_feature_set("T0", "S1"), get_feature_set("T1", "S1")
    expected, expected_steps = cache(step_graph, snapshot, t0, cache_root=tmp_path)
    cache(step_graph, snapshot, t1, cache_root=tmp_path)
    without_last_edge = EllipticDataset(
        step_graph.transactions, step_graph.edges.iloc[:-1]
    )
    cache(without_last_edge, fingerprint(without_last_edge), t0, cache_root=tmp_path)
    cache(step_graph, snapshot, get_feature_set("T0", "S2"), cache_root=tmp_path)
    assert len(list(tmp_path.glob("*/*/manifest.json"))) == 4

    def unexpected_build(*args: object) -> None:
        pytest.fail("A cached or prohibited development window reached the builder.")

    monkeypatch.setattr(feature_cache, "build_feature_set", unexpected_build)
    actual, steps = cache(step_graph, snapshot, t0, cache_root=tmp_path)
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    pd.testing.assert_series_equal(expected_steps, steps)
    future = EllipticDataset(
        pd.concat([step_graph.transactions, _transactions(["8"], 40)]), step_graph.edges
    )
    with pytest.raises(DatasetError, match="development transactions"):
        cache(future, snapshot, t0, cache_root=tmp_path)


def test_feature_cache_verifies_predictor_values_before_reuse(
    step_graph: EllipticDataset, tmp_path: Path
) -> None:
    feature_set = get_feature_set("T2", "S2")
    fingerprint = feature_cache.development_snapshot_digest
    cache = feature_cache.cached_development_features
    snapshot = fingerprint(step_graph)
    original, steps = cache(step_graph, snapshot, feature_set, cache_root=tmp_path)
    modified = EllipticDataset(step_graph.transactions.copy(), step_graph.edges)
    modified.transactions.loc[0, "feature_001"] = np.nextafter(
        modified.transactions.loc[0, "feature_001"], np.inf
    )
    with pytest.raises(DatasetError, match="SHA-256 does not match"):
        cache(modified, snapshot, feature_set, cache_root=tmp_path)
    modified_snapshot = fingerprint(modified)
    assert modified_snapshot != snapshot
    rebuilt, rebuilt_steps = cache(
        modified, modified_snapshot, feature_set, cache_root=tmp_path
    )
    assert original.iloc[0, 0] != rebuilt.iloc[0, 0]
    pd.testing.assert_series_equal(steps, rebuilt_steps)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_feature_cache_rebuilds_missing_or_corrupt_payloads(
    damage: str, step_graph: EllipticDataset, tmp_path: Path
) -> None:
    snapshot = feature_cache.development_snapshot_digest(step_graph)
    feature_set = get_feature_set("T0", "S1")
    cache = feature_cache.cached_development_features
    expected, expected_steps = cache(
        step_graph, snapshot, feature_set, cache_root=tmp_path
    )
    (payload,) = tmp_path.glob("*/*/features.joblib")
    if damage == "missing":
        payload.unlink()
    else:
        payload.write_bytes(b"corrupt")
    actual, actual_steps = cache(step_graph, snapshot, feature_set, cache_root=tmp_path)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_series_equal(actual_steps, expected_steps)
    manifest = read_json(payload.with_name("manifest.json"))
    assert manifest["cache_sha256"] == file_digest(payload)


def test_feature_cache_uses_an_explicit_root_across_working_directories(
    step_graph: EllipticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    root = tmp_path / "project" / "data" / "processed"
    snapshot = feature_cache.development_snapshot_digest(step_graph)
    feature_set = get_feature_set("T0", "S1")
    cache = feature_cache.cached_development_features
    monkeypatch.chdir(first)
    expected, _ = cache(step_graph, snapshot, feature_set, cache_root=root)
    monkeypatch.chdir(second)
    actual, _ = cache(step_graph, snapshot, feature_set, cache_root=root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert len(list(root.glob("*/*/manifest.json"))) == 1
    assert not any(first.iterdir()) and not any(second.iterdir())
    with pytest.raises(DatasetError, match="absolute path"):
        cache(step_graph, snapshot, feature_set, cache_root=Path("data/processed"))
