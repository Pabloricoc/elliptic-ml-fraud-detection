from pathlib import Path

import pytest

from elliptic_fraud import development
from elliptic_fraud.feature_cache import development_snapshot_digest
from elliptic_fraud.runs import file_digest, snapshot_digest
from elliptic_fraud.schema import EXPECTED_FILES, DatasetError


def test_export_keeps_only_steps_1_to_39_and_records_original_file_digests(
    model_input: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "development"
    exporter = development.load_dataset
    reads = []

    def load_once(directory: Path):
        reads.append(directory)
        return exporter(directory)

    monkeypatch.setattr(development, "load_dataset", load_once)
    assert development.export_development(model_input, destination) == destination
    export, manifest = development.load_development_export(destination)
    assert set(export.transactions["time_step"]) == {1, 29, 30, 39}
    assert set(export.transactions["class"]) == {"1", "2", "unknown"}
    ids = export.transactions["tx_id"]
    assert export.edges.isin(set(ids)).all(axis=None)
    assert len(export.edges) == 12
    assert manifest["steps"] == [1, 39]
    files = {name: file_digest(model_input / name) for name in EXPECTED_FILES}
    assert manifest["files_sha256"] == files
    assert manifest["source_snapshot_sha256"] == snapshot_digest(files)
    assert manifest["development_snapshot_sha256"] == development_snapshot_digest(
        export
    )

    with pytest.raises(DatasetError, match="already exists"):
        development.export_development(model_input, destination)
    assert reads == [model_input]
