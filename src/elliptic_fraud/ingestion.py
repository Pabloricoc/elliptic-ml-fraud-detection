"""Read the original CSVs into validated transaction and directed-edge tables."""

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    CLASS_NAMES,
    CLASS_SCHEMA,
    CLASSES_FILE,
    EDGE_SCHEMA,
    EDGES_FILE,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    FEATURES_FILE,
    DatasetError,
    missing_files,
)
from .validation import validate_graph, validate_tables


@dataclass
class EllipticDataset:
    """One row per transaction in source order, plus the separate directed edge list."""

    transactions: pd.DataFrame
    edges: pd.DataFrame


def _text_lines(stream: Iterable[str], file: str) -> Iterator[str]:
    # pandas' C parser can silently truncate a field at a NUL byte.
    for line_number, line in enumerate(stream, start=1):
        if "\x00" in line:
            raise DatasetError(f"{file}: NUL byte on line {line_number}.")
        yield line


def _read_csv(
    path: Path,
    columns: tuple[str, ...],
    header: tuple[str, ...] | None,
    *,
    usecols: list[int] | None = None,
) -> pd.DataFrame:
    # Text keys preserve leading zeros and identifiers beyond floating-point precision.
    selected = list(range(len(columns))) if usecols is None else usecols
    dtypes = {
        i: "float64" if columns[i] in FEATURE_COLUMNS else "string" for i in selected
    }
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(_text_lines(stream, path.name), strict=True)
            if header is not None and next(reader, None) != list(header):
                raise DatasetError(f"{path.name}: expected header {','.join(header)}.")
            first = next(reader, None)
            if first is None:
                return pd.DataFrame(
                    {columns[i]: pd.Series(dtype=dtypes[i]) for i in selected}
                )
            if (
                header is None
                and first
                and not (first[0].isascii() and first[0].isdecimal())
            ):
                raise DatasetError(
                    f"{path.name}: expected headerless data starting with a transaction ID."
                )
            for row in chain((first,), reader):
                if len(row) != len(columns):
                    raise DatasetError(
                        f"{path.name}: expected {len(columns)} fields per row, "
                        f"found {len(row)} at line {reader.line_num}."
                    )
            stream.seek(0)
            # Infer the CSV width from data. Supplying names here can make pandas
            # silently interpret a surplus leading field as an index.
            frame = pd.read_csv(
                stream,
                header=None,
                skiprows=int(header is not None),
                dtype=dtypes,
                usecols=usecols,
                skip_blank_lines=False,
                float_precision="round_trip",
            )
    except DatasetError:
        raise
    except FileNotFoundError:
        raise DatasetError(f"Missing input file: {path}") from None
    except OSError as error:
        raise DatasetError(f"Cannot read {path}: {error.strerror}") from None
    except (csv.Error, UnicodeError, pd.errors.ParserError):
        raise DatasetError(
            f"{path.name}: malformed CSV or invalid UTF-8 encoding."
        ) from None
    except (ValueError, OverflowError):
        # Parser messages can contain raw transaction values.
        raise DatasetError(
            f"{path.name}: invalid value during typed CSV parsing. "
            "Numeric feature fields must contain finite numbers."
        ) from None
    if frame.shape[1] != len(selected):
        raise DatasetError(f"{path.name}: expected {len(columns)} fields per row.")
    frame.columns = [columns[i] for i in selected]
    return frame


def load_scoring_graph(directory: Path) -> EllipticDataset:
    """Read transaction IDs, time steps and edges without opening the class table."""
    nodes = _read_csv(directory / FEATURES_FILE, FEATURE_SCHEMA, None, usecols=[0, 1])
    edges = _read_csv(directory / EDGES_FILE, EDGE_SCHEMA, ("txId1", "txId2"))
    validate_graph(nodes, edges)
    nodes["time_step"] = nodes["time_step"].astype(np.uint8)
    return EllipticDataset(nodes, edges)


def load_dataset(directory: Path = Path("data/raw")) -> EllipticDataset:
    """Read and validate a local release, joining class codes by exact transaction ID."""
    missing = missing_files(directory)
    if missing:
        raise DatasetError(f"Missing input files in {directory}: {', '.join(missing)}.")
    features = _read_csv(directory / FEATURES_FILE, FEATURE_SCHEMA, None)
    classes = _read_csv(directory / CLASSES_FILE, CLASS_SCHEMA, ("txId", "class"))
    edges = _read_csv(directory / EDGES_FILE, EDGE_SCHEMA, ("txId1", "txId2"))
    validate_tables(features, classes, edges)
    features["time_step"] = features["time_step"].astype(np.uint8)
    labels = classes.set_index("tx_id")["class"]
    features["class"] = pd.Categorical(
        features["tx_id"].map(labels), categories=list(CLASS_NAMES)
    )
    return EllipticDataset(features, edges)
