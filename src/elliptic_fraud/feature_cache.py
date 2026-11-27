"""Cache versioned predictor matrices built from development-only inputs."""

from pathlib import Path

import joblib
import pandas as pd
from filelock import FileLock

from .folds import DEV_STEPS
from .ingestion import EllipticDataset
from .model_features import FeatureSet, build_feature_set, get_feature_set
from .runs import (
    file_digest,
    implementation_digest,
    read_json,
    snapshot_digest,
    write_json,
)
from .schema import DatasetError


def cached_development_features(
    dataset: EllipticDataset, snapshot_sha: str, feature_set: FeatureSet
) -> tuple[pd.DataFrame, pd.Series]:
    """Reuse an exact feature contract and source fingerprint for steps 1–39."""
    if len(snapshot_sha) != 64 or any(
        char not in "0123456789abcdef" for char in snapshot_sha
    ):
        raise DatasetError("Expected a SHA-256 snapshot identifier.")
    steps = dataset.transactions.set_index("tx_id")["time_step"]
    if (
        steps.empty
        or not steps.index.is_unique
        or not steps.between(*DEV_STEPS).all()
        or not dataset.edges[["source_id", "target_id"]]
        .isin(steps.index)
        .all(axis=None)
    ):
        raise DatasetError(
            "Feature caching requires only aligned development transactions and edges from steps 1–39."
        )
    # Compose only registered tiers, so identical manifests cannot name different builders.
    registered = get_feature_set(
        feature_set.name, feature_set.scenario, feature_set.version
    )
    if registered.columns != feature_set.columns:
        raise DatasetError("The cache feature order must match the registry.")
    config = {
        "snapshot_sha256": snapshot_sha,
        "feature_set": registered.manifest(),
        "implementation_sha256": implementation_digest(),
    }
    root = Path("data/processed").resolve()
    directory = (root / snapshot_sha / f"features-{snapshot_digest(config)}").resolve()
    if not directory.is_relative_to(root):
        raise DatasetError(
            "Feature caches must be inside the processed-data directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / ".cache.lock"):
        manifest = directory / "manifest.json"
        payload = directory / "features.joblib"
        if manifest.exists():
            stored = read_json(manifest)
            if stored.get("config") != config or file_digest(payload) != stored.get(
                "cache_sha256"
            ):
                raise DatasetError("The feature cache does not match its manifest.")
            features, cached_steps = joblib.load(payload)
            if (
                not cached_steps.equals(steps)
                or not features.index.equals(steps.index)
                or tuple(features.columns) != registered.columns
            ):
                raise DatasetError(
                    "The feature cache does not match the requested development window."
                )
            return features, cached_steps
        features, cached_steps = build_feature_set(dataset, registered, *DEV_STEPS)
        temporary = directory / ".features.partial"
        joblib.dump((features, cached_steps), temporary, compress=3)
        temporary.replace(payload)
        write_json(
            directory / ".manifest.partial",
            {"config": config, "cache_sha256": file_digest(payload)},
        )
        (directory / ".manifest.partial").replace(manifest)
        return features, cached_steps
