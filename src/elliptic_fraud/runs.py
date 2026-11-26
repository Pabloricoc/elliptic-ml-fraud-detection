"""Publish complete local runs with the access policy of their output directory."""

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .schema import DatasetError


def file_digest(path: Path) -> str:
    """Hash file contents without loading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_state(paths: list[Path]) -> list[tuple[int, int]]:
    """Capture size and modification time for input consistency checks."""
    stats = (path.stat() for path in paths)
    return [(stat.st_size, stat.st_mtime_ns) for stat in stats]


def snapshot_digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()


def implementation_digest() -> str:
    return snapshot_digest(
        {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")}
    )


def write_json(path: Path, values: dict) -> None:
    path.write_text(
        json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise DatasetError(f"Invalid JSON in {path.name}.") from error
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a JSON object in {path.name}.")
    return value


@contextmanager
def new_run(output_dir: Path, prefix: str) -> Iterator[tuple[Path, Path]]:
    """Yield staging and final paths. Publish by rename after all writes succeed."""
    parent = output_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{prefix}-{uuid4().hex}"
    if destination.parent != parent:
        raise ValueError("Run prefix must be a single filename component.")
    staging = parent / f".{destination.name}.partial"
    # An ordinary sibling inherits the parent's ACL on Windows. A directory
    # moved out of TemporaryDirectory can retain its owner-only ACL.
    staging.mkdir()
    try:
        (staging / ".gitignore").write_text("*\n", encoding="utf-8")
        yield staging, destination
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
