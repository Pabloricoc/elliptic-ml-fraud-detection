"""Small generated CSVs with the published widths, headers and label vocabulary."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.schema import CLASSES_FILE, EDGES_FILE, FEATURES_FILE


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw files"
    directory.mkdir()
    # Deliberately shuffled labels and time steps. No values come from Elliptic.
    rows = (
        ("9007199254740993", 3, 4.0),
        ("0007", 1, 1.0),
        ("9007199254740992", 1, 2.0),
        ("0088", 1, 3.0),
    )
    with (directory / "elliptic_txs_features.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for tx_id, step, signal in rows:
            writer.writerow([tx_id, step, signal, *range(2, 166)])
    (directory / "elliptic_txs_classes.csv").write_text(
        "txId,class\n0088,unknown\n9007199254740992,1\n9007199254740993,unknown\n0007,2\n",
        encoding="utf-8-sig",
    )
    (directory / "elliptic_txs_edgelist.csv").write_text(
        "txId1,txId2\n0007,9007199254740992\n0088,9007199254740992\n", encoding="utf-8"
    )
    return directory


@pytest.fixture
def periods(request: pytest.FixtureRequest) -> EllipticDataset:
    # Stand-alone invented graphs straddle every split boundary.
    rows, edges = [], []
    for step in getattr(request, "param", (1, 29, 30, 39, 40, 49)):
        a, b, c, d = [f"000000{step:02}{i}" for i in range(4)]
        rows.extend(
            [(a, step, "unknown"), (b, step, "2"), (c, step, "1"), (d, step, "2")]
        )
        edges.extend([(a, b), (b, c), (d, c)])
    return EllipticDataset(
        pd.DataFrame(rows, columns=["tx_id", "time_step", "class"]),
        pd.DataFrame(edges, columns=["source_id", "target_id"]),
    )


@pytest.fixture
def model_input(tmp_path: Path, periods: EllipticDataset) -> Path:
    directory = tmp_path / "invented inputs"
    directory.mkdir()
    with (directory / FEATURES_FILE).open("w", newline="") as stream:
        writer = csv.writer(stream)
        for row in periods.transactions.itertuples(index=False, name=None):
            writer.writerow([row[0], row[1], *range(165)])
    periods.transactions[["tx_id", "class"]].to_csv(
        directory / CLASSES_FILE, header=["txId", "class"], index=False
    )
    periods.edges.to_csv(directory / EDGES_FILE, header=["txId1", "txId2"], index=False)
    return directory
