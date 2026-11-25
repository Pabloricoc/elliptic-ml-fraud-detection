"""Blocking checks on the three parsed tables, before attaching labels."""

import numpy as np
import pandas as pd

from .schema import (
    CLASS_NAMES,
    CLASSES_FILE,
    EDGES_FILE,
    FEATURE_COLUMNS,
    FEATURES_FILE,
    TIME_STEPS,
    DatasetError,
)


def _check_ids(ids: pd.Series, file: str, *, unique: bool = False) -> None:
    if (
        not pd.api.types.is_string_dtype(ids.dtype)
        or not ids.str.fullmatch(r"[0-9]+", na=False).all()
    ):
        raise DatasetError(f"{file}: {ids.name} requires nonempty decimal identifiers.")
    if unique and ids.duplicated().any():
        raise DatasetError(f"{file}: duplicate transaction identifiers in {ids.name}.")


def validate_tables(
    features: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame
) -> None:
    """Validate parsed input frames with canonical column names and text keys.

    All feature IDs must have exactly one explicit class, including unknown.
    The original graph consists of disjoint time-step snapshots.
    """
    if features.empty or classes.empty:
        raise DatasetError(
            "The feature and class tables must both contain transactions."
        )
    _check_ids(features["tx_id"], FEATURES_FILE, unique=True)
    _check_ids(classes["tx_id"], CLASSES_FILE, unique=True)

    if not features["time_step"].isin([str(step) for step in TIME_STEPS]).all():
        raise DatasetError(
            f"{FEATURES_FILE}: time_step must be an integer from 1 to 49."
        )
    for name in FEATURE_COLUMNS:
        column = features[name]
        if (
            not pd.api.types.is_numeric_dtype(column)
            or not np.isfinite(column.to_numpy(copy=False)).all()
        ):
            raise DatasetError(
                f"{FEATURES_FILE}: {name} requires finite numeric values."
            )

    if not classes["class"].isin(CLASS_NAMES).all():
        raise DatasetError(f"{CLASSES_FILE}: class must be 1, 2 or unknown.")
    missing = (~features["tx_id"].isin(classes["tx_id"])).sum()
    extra = (~classes["tx_id"].isin(features["tx_id"])).sum()
    if missing or extra:
        raise DatasetError(
            f"Transaction ID mismatch: {missing} feature rows lack a class row, "
            f"{extra} class rows lack a feature row."
        )

    for name in ("source_id", "target_id"):
        _check_ids(edges[name], EDGES_FILE)
    steps = features[["tx_id", "time_step"]].set_index("tx_id")["time_step"]
    source_step = edges["source_id"].map(steps)
    target_step = edges["target_id"].map(steps)
    for name, mapped in (("source_id", source_step), ("target_id", target_step)):
        if mapped.isna().any():
            raise DatasetError(f"{EDGES_FILE}: {name} references absent transactions.")
    if source_step.ne(target_step).any():
        raise DatasetError(f"{EDGES_FILE}: an edge connects different time steps.")
