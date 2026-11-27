"""Create and verify the local development-only export of the Elliptic release."""

import sys
from pathlib import Path

import joblib
from filelock import FileLock

from .feature_cache import development_snapshot_digest
from .folds import DEV_STEPS
from .ingestion import EllipticDataset, load_dataset
from .runs import (
    file_digest,
    file_state,
    new_run,
    read_json,
    snapshot_digest,
    write_json,
)
from .schema import EXPECTED_FILES, DatasetError

DEVELOPMENT_DIR = Path("data/processed/development")


def _restrict_to_development(dataset: EllipticDataset) -> EllipticDataset:
    transactions = dataset.transactions.loc[
        dataset.transactions["time_step"].between(*DEV_STEPS)
    ].copy()
    ids = transactions["tx_id"]
    edges = dataset.edges.loc[
        dataset.edges["source_id"].isin(ids) & dataset.edges["target_id"].isin(ids)
    ].copy()
    return EllipticDataset(transactions, edges)


def export_development(raw_dir: Path, destination: Path) -> Path:
    """Publish steps 1–39 from the one approved source read. Refuse to replace an export."""
    destination = destination.absolute()
    protected = [
        raw_dir,
        *(Path("artifacts") / name for name in ("models", "scores", "eda", "tracking")),
    ]
    if any(destination.resolve().is_relative_to(path.resolve()) for path in protected):
        raise DatasetError(
            "Choose an export directory outside raw data and protected artifacts."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(destination.with_name(f".{destination.name}.lock")):
        if destination.exists() or destination.is_symlink():
            raise DatasetError(
                "The development export already exists. Use the existing export."
            )
        paths = [raw_dir / name for name in EXPECTED_FILES]
        before = file_state(paths)
        development = _restrict_to_development(load_dataset(raw_dir))
        fingerprint = {path.name: file_digest(path) for path in paths}
        if file_state(paths) != before:
            raise DatasetError(
                "The release files changed during the development export."
            )

        with new_run(destination.parent, "development-export") as (staging, completed):
            payload = staging / "development.joblib"
            joblib.dump(development, payload, compress=3)
            write_json(
                staging / "manifest.json",
                {
                    "schema_version": 1,
                    "steps": DEV_STEPS,
                    "files_sha256": fingerprint,
                    "source_snapshot_sha256": snapshot_digest(fingerprint),
                    "development_snapshot_sha256": development_snapshot_digest(
                        development
                    ),
                    "export_sha256": file_digest(payload),
                },
            )
        completed.rename(destination)
    return destination


def load_development_export(directory: Path) -> tuple[EllipticDataset, dict]:
    """Load only a verified development export. Raw CSVs are never used here."""
    manifest = read_json(directory / "manifest.json")
    fingerprint = manifest.get("files_sha256")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("steps") != list(DEV_STEPS)
        or not isinstance(fingerprint, dict)
        or set(fingerprint) != set(EXPECTED_FILES)
        or snapshot_digest(fingerprint) != manifest.get("source_snapshot_sha256")
    ):
        raise DatasetError("Invalid development export manifest or source fingerprint.")
    payload = directory / "development.joblib"
    if file_digest(payload) != manifest.get("export_sha256"):
        raise DatasetError("The development export does not match its payload digest.")
    dataset = joblib.load(payload)
    if not isinstance(dataset, EllipticDataset) or development_snapshot_digest(
        dataset
    ) != manifest.get("development_snapshot_sha256"):
        raise DatasetError("The development frames do not match their manifest.")
    return dataset, manifest


if __name__ == "__main__":
    try:
        print(
            f"Development export written to {export_development(Path('data/raw'), DEVELOPMENT_DIR)}"
        )
    except (DatasetError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
