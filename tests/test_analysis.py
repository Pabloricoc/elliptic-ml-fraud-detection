import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pandas as pd
import pytest

from elliptic_fraud.analysis import feature_statistics, summarize
from elliptic_fraud.data_cli import main
from elliptic_fraud.ingestion import load_dataset


def test_coverage_known_class_balance_and_directional_counts(raw_dir: Path) -> None:
    data = load_dataset(raw_dir)
    report, periods = summarize(data)
    assert report.class_counts == {"illicit": 1, "licit": 1, "unknown": 2}
    assert report.unknown_fraction == 0.5
    assert report.known_class_shares == {"illicit": 0.5, "licit": 0.5}
    assert periods.index.tolist() == list(range(1, 50))
    assert periods.loc[1, "label_coverage"] == pytest.approx(2 / 3)
    assert periods.loc[1, "illicit_share_labelled"] == 0.5
    assert periods.loc[3, "label_coverage"] == 0
    assert pd.isna(periods.loc[3, "illicit_share_labelled"])
    assert periods.loc[2, "transactions"] == 0
    assert pd.isna(periods.loc[2, "label_coverage"])
    assert report.graph == {
        "nodes": 4,
        "directed_edges": 2,
        "isolated_nodes": 1,
        "duplicate_edge_rows": 0,
        "self_loops": 0,
        "max_in_degree": 2,
        "max_out_degree": 1,
    }
    assert "feature_001" not in report.constant_features
    assert len(report.constant_features) == 164


def test_all_unknown_single_transaction_and_header_only_graph(raw_dir: Path) -> None:
    features = raw_dir / "elliptic_txs_features.csv"
    features.write_text(features.read_text().splitlines()[0] + "\n")
    (raw_dir / "elliptic_txs_classes.csv").write_text(
        "txId,class\n9007199254740993,unknown\n"
    )
    (raw_dir / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n")
    data = load_dataset(raw_dir)
    summary, _ = summarize(data)
    assert summary.known_class_shares == {"illicit": None, "licit": None}
    assert summary.unknown_fraction == 1
    assert summary.graph["isolated_nodes"] == 1
    assert summary.graph["max_in_degree"] == summary.graph["max_out_degree"] == 0
    assert pd.isna(feature_statistics(data.transactions).loc["feature_001", "std"])


def test_duplicate_edges_and_self_loops_are_retained_and_flagged(raw_dir: Path) -> None:
    with (raw_dir / "elliptic_txs_edgelist.csv").open("a") as stream:
        stream.write("0007,9007199254740992\n0007,0007\n")
    data = load_dataset(raw_dir)
    summary, _ = summarize(data)
    assert len(data.edges) == 4
    assert summary.graph["duplicate_edge_rows"] == summary.graph["self_loops"] == 1
    assert any("self-loops" in w for w in summary.warnings)


def test_installed_cli_publishes_only_aggregates(raw_dir: Path, tmp_path: Path) -> None:
    command = shutil.which("elliptic-data", path=sysconfig.get_path("scripts"))
    assert command is not None
    output = tmp_path / "private reports"
    result = subprocess.run(
        [command, "analyze", "--data-dir", str(raw_dir), "--output-dir", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (run,) = output.iterdir()
    assert str(run) in result.stdout
    assert {p.name for p in run.iterdir()} == {
        "summary.json",
        "time_steps.csv",
        "features.csv",
        "timeline.png",
        ".gitignore",
    }
    summary = json.loads((run / "summary.json").read_text())
    assert summary["shapes"] == {
        "features": [4, 167],
        "classes": [4, 2],
        "edges": [2, 2],
        "joined_transactions": [4, 168],
    }
    table = pd.read_csv(run / "features.csv").set_index("feature")
    assert len(table) == 165
    assert table.loc["feature_001", ["count", "median", "q25", "q75"]].tolist() == [
        4,
        2.5,
        1.75,
        3.25,
    ]
    assert pd.read_csv(run / "time_steps.csv")["transactions"].sum() == 4
    assert (run / "timeline.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for filename in ("summary.json", "time_steps.csv", "features.csv"):
        assert "9007199254740993" not in (run / filename).read_text()


def test_validate_uses_default_directory_without_creating_reports(
    raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "data").mkdir()
    raw_dir.rename(tmp_path / "data" / "raw")
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 0
    captured = capsys.readouterr()
    assert "4 transactions" in captured.out and "Warning" in captured.err
    assert not (tmp_path / "artifacts").exists()


def test_invalid_dataset_leaves_previous_reports_untouched(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports"
    output.mkdir()
    previous = output / "previous.txt"
    previous.write_text("keep")
    (raw_dir / "elliptic_txs_classes.csv").unlink()
    assert (
        main(["analyze", "--data-dir", str(raw_dir), "--output-dir", str(output)]) == 1
    )
    assert "Missing input file" in capsys.readouterr().err
    assert list(output.iterdir()) == [previous] and previous.read_text() == "keep"


def test_preserves_raw_directory_and_cleans_failed_report(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        main(
            [
                "analyze",
                "--data-dir",
                str(raw_dir),
                "--output-dir",
                str(raw_dir / "report"),
            ]
        )
        == 1
    )
    assert not (raw_dir / "report").exists()

    def fail_plot(periods: pd.DataFrame, path: Path) -> None:
        raise OSError("test write failure")

    monkeypatch.setattr("elliptic_fraud.report._plot_timeline", fail_plot)
    output = tmp_path / "reports"
    assert (
        main(["analyze", "--data-dir", str(raw_dir), "--output-dir", str(output)]) == 1
    )
    assert not list(output.iterdir())
