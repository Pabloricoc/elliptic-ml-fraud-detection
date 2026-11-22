import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from elliptic_fraud.check_data import main


def test_installed_command_finds_the_default_layout(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for name in (
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_features.csv",
    ):
        (raw / name).touch()

    command = shutil.which("elliptic-check-data", path=sysconfig.get_path("scripts"))
    assert command is not None, "Editable installation must provide elliptic-check-data"
    result = subprocess.run([command], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "3 expected filenames" in result.stdout
    assert str(Path("data", "raw")) in result.stdout
    assert result.stderr == ""


def test_partial_layout_reports_every_missing_regular_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "manual extraction"
    directory.mkdir()
    (directory / "elliptic_txs_classes.csv").touch()
    (directory / "elliptic_txs_edgelist.csv").mkdir()

    assert main([str(directory)]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    header, *missing = output.err.splitlines()
    assert str(directory) in header
    assert set(missing) == {"elliptic_txs_edgelist.csv", "elliptic_txs_features.csv"}


def test_absent_directory_is_reported_and_left_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "raw"

    assert main([str(directory)]) == 1

    assert str(directory) in capsys.readouterr().err
    assert not directory.exists()


def test_file_passed_as_directory_explains_the_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "download.zip"
    archive.touch()

    assert main([str(archive)]) == 1

    message = capsys.readouterr().err
    assert "directory" in message
    assert str(archive) in message
