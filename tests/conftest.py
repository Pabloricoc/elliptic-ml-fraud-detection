"""Small generated CSVs with the published widths, headers and label vocabulary."""

import csv
from pathlib import Path

import pytest


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
