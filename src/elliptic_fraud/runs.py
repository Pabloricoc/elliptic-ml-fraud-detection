"""Publish complete local runs with the access policy of their output directory."""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


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
