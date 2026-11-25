"""File and column contracts for the original 2019 Elliptic release."""

from pathlib import Path

CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"
FEATURES_FILE = "elliptic_txs_features.csv"
EXPECTED_FILES = (CLASSES_FILE, EDGES_FILE, FEATURES_FILE)

# The release counts the time step among its 166 features.
FEATURE_COLUMNS = tuple(f"feature_{n:03d}" for n in range(1, 166))
FEATURE_SCHEMA = ("tx_id", "time_step", *FEATURE_COLUMNS)
CLASS_SCHEMA = ("tx_id", "class")
EDGE_SCHEMA = ("source_id", "target_id")
CLASS_NAMES = {"1": "illicit", "2": "licit", "unknown": "unknown"}
TIME_STEPS = range(1, 50)


class DatasetError(ValueError):
    """A local input violates the original CSV or relational contract."""


def missing_files(directory: Path) -> list[str]:
    """List required filenames that do not resolve to regular files."""
    return [name for name in EXPECTED_FILES if not (directory / name).is_file()]
