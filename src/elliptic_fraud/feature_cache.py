"""Cache versioned predictor matrices built from verified development inputs."""

import hashlib
import json
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


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(
        json.dumps(
            list(zip(frame.columns, frame.dtypes.astype(str), strict=True))
        ).encode("utf-8")
    )
    for _, column in frame.items():
        values = column.to_numpy()
        if values.dtype.kind in "biuf":
            # Preserve all numeric bits independently of the host byte order.
            content = values.astype(
                values.dtype.newbyteorder("<"), copy=False
            ).tobytes()
        else:
            content = column.astype("string").to_json(orient="values").encode("utf-8")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _development_snapshot(dataset: EllipticDataset) -> tuple[str, pd.Series]:
    steps = dataset.transactions[["tx_id", "time_step"]].set_index("tx_id")["time_step"]
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
    digest = snapshot_digest(
        {
            "transactions": _frame_digest(dataset.transactions),
            "edges": _frame_digest(dataset.edges),
        }
    )
    return digest, steps


def development_snapshot_digest(dataset: EllipticDataset) -> str:
    """Fingerprint the development frames, including their values, dtypes and order."""
    return _development_snapshot(dataset)[0]


def cached_development_features(
    dataset: EllipticDataset,
    snapshot_sha: str,
    feature_set: FeatureSet,
    *,
    cache_root: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reuse predictors for a verified development snapshot under an explicit root."""
    if not cache_root.is_absolute():
        raise DatasetError("The feature cache root must be an absolute path.")
    verified_sha, steps = _development_snapshot(dataset)
    if snapshot_sha != verified_sha:
        raise DatasetError(
            "The development snapshot SHA-256 does not match the supplied data."
        )
    # Compose only registered tiers, so identical manifests cannot name different builders.
    registered = get_feature_set(
        feature_set.name, feature_set.scenario, feature_set.version
    )
    if registered.columns != feature_set.columns:
        raise DatasetError("The cache feature order must match the registry.")
    config = {
        "snapshot_sha256": verified_sha,
        "feature_set": registered.manifest(),
        "implementation_sha256": implementation_digest(),
    }
    root = cache_root.resolve()
    directory = (root / verified_sha / f"features-{snapshot_digest(config)}").resolve()
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
            if stored.get("config") != config:
                raise DatasetError("The feature cache does not match its manifest.")
            try:
                digest = file_digest(payload)
            except FileNotFoundError:
                digest = None
            if digest is not None and digest == stored.get("cache_sha256"):
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
