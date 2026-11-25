import csv
from pathlib import Path

import numpy as np
import pytest

from elliptic_fraud.ingestion import load_dataset
from elliptic_fraud.schema import DatasetError

FEATURES = "elliptic_txs_features.csv"
CLASSES = "elliptic_txs_classes.csv"
EDGES = "elliptic_txs_edgelist.csv"


def _change_cell(path: Path, row: int, column: int, value: str) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[row][column] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_load_preserves_first_row_keys_labels_features_and_direction(
    raw_dir: Path,
) -> None:
    originals = {p: p.read_bytes() for p in raw_dir.iterdir()}
    dataset = load_dataset(raw_dir)
    tx = dataset.transactions
    assert tx.shape == (4, 168)
    assert tx["tx_id"].tolist() == [
        "9007199254740993",
        "0007",
        "9007199254740992",
        "0088",
    ]
    assert tx["time_step"].tolist() == [3, 1, 1, 1]
    assert tx["time_step"].dtype == np.uint8
    assert tx["class"].tolist() == ["unknown", "2", "1", "unknown"]
    assert tx["feature_001"].tolist() == [4.0, 1.0, 2.0, 3.0]
    assert tx["feature_165"].eq(165).all()
    assert tx.filter(like="feature_").dtypes.eq(np.dtype("float64")).all()
    assert list(dataset.edges.itertuples(index=False, name=None)) == [
        ("0007", "9007199254740992"),
        ("0088", "9007199254740992"),
    ]
    assert all(p.read_bytes() == contents for p, contents in originals.items())


@pytest.mark.parametrize("file", [FEATURES, CLASSES, EDGES])
def test_missing_inputs_name_the_file(raw_dir: Path, file: str) -> None:
    (raw_dir / file).unlink()
    with pytest.raises(DatasetError, match=file):
        load_dataset(raw_dir)


@pytest.mark.parametrize(
    ("file", "content", "message"),
    [
        (CLASSES, "1,unknown\n", "expected header"),
        (EDGES, "txId2,txId1\n", "expected header"),
        (CLASSES, "txId,class,other\n", "expected header"),
        (FEATURES, "id,step," + ",".join(["value"] * 165), "headerless"),
        (FEATURES, "42,1,3\n", "167 fields"),
        (EDGES, "txId1,txId2\n0007,0088,0007\n", "2 fields"),
        (CLASSES, "txId,class\n0007,1\n0088,unknown,surplus\n", "2 fields"),
        (CLASSES, "txId,class\n", "contain transactions"),
        (FEATURES, "", "contain transactions"),
        (EDGES, "", "expected header"),
    ],
)
def test_header_and_row_contract(
    raw_dir: Path, file: str, content: str, message: str
) -> None:
    (raw_dir / file).write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(raw_dir)


def test_truncated_feature_row_is_not_silently_padded(raw_dir: Path) -> None:
    path = raw_dir / FEATURES
    with path.open("a", encoding="utf-8") as stream:
        stream.write("42,2,5\n")
    with pytest.raises(DatasetError, match="finite numeric|167 fields"):
        load_dataset(raw_dir)


@pytest.mark.parametrize("bad_bytes", [b"\xff", b'"unterminated'])
def test_corrupt_csv_has_a_file_specific_error(raw_dir: Path, bad_bytes: bytes) -> None:
    (raw_dir / FEATURES).write_bytes(bad_bytes)
    with pytest.raises(DatasetError, match=FEATURES):
        load_dataset(raw_dir)


@pytest.mark.parametrize(
    ("file", "row", "column", "prefix"),
    [
        (FEATURES, 2, -1, "165"),
        (CLASSES, 2, 0, "9007199254740992"),
        (EDGES, 1, 0, "0007"),
    ],
)
def test_embedded_nul_cannot_silently_truncate_a_value(
    raw_dir: Path, file: str, row: int, column: int, prefix: str
) -> None:
    _change_cell(raw_dir / file, row, column, prefix + "\x00hidden")
    with pytest.raises(DatasetError, match="NUL byte"):
        load_dataset(raw_dir)


@pytest.mark.parametrize(
    ("file", "row", "col", "value", "message"),
    [
        (FEATURES, 1, 0, "9007199254740993", "duplicate transaction"),
        (CLASSES, 1, 0, "0007", "duplicate transaction"),
        (FEATURES, 1, 0, "", "decimal identifiers"),
        (CLASSES, 1, 0, "7.0", "decimal identifiers"),
        (EDGES, 1, 1, "", "decimal identifiers"),
        (CLASSES, 2, 1, "0", "class must be"),
        (CLASSES, 1, 1, "", "class must be"),
        (CLASSES, 1, 1, "UNKNOWN", "class must be"),
        (CLASSES, 1, 0, "123", "1 feature rows lack a class row, 1 class rows lack"),
        (EDGES, 1, 0, "404", "source_id references absent"),
        (EDGES, 1, 1, "404", "target_id references absent"),
        (EDGES, 1, 1, "9007199254740993", "different time steps"),
    ],
)
def test_invalid_keys_labels_and_edges(
    raw_dir: Path, file: str, row: int, col: int, value: str, message: str
) -> None:
    _change_cell(raw_dir / file, row, col, value)
    with pytest.raises(DatasetError, match=message):
        load_dataset(raw_dir)


@pytest.mark.parametrize("step", ["0", "50", "-1", "1.5", "", "nan"])
def test_invalid_time_steps(raw_dir: Path, step: str) -> None:
    _change_cell(raw_dir / FEATURES, 1, 1, step)
    with pytest.raises(DatasetError, match="time_step"):
        load_dataset(raw_dir)


@pytest.mark.parametrize("value", ["NaN", "", "inf", "-inf", "not-a-number"])
def test_missing_nonfinite_and_nonnumeric_features(raw_dir: Path, value: str) -> None:
    _change_cell(raw_dir / FEATURES, 1, -1, value)
    with pytest.raises(DatasetError, match="finite numeric|invalid value"):
        load_dataset(raw_dir)
