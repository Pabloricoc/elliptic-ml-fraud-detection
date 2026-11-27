import os
import shutil
import subprocess
import sysconfig
from dataclasses import replace
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from elliptic_fraud import experiments
from elliptic_fraud.experiments import (
    TRACKING_EXPERIMENT,
    cache_development,
    development_cache_path,
    run_baseline_folds,
)
from elliptic_fraud.folds import development_folds, slice_development
from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.model_features import build_window
from elliptic_fraud.runs import read_json
from elliptic_fraud.schema import DatasetError


def test_fixed_expanding_folds_and_optional_embargo() -> None:
    folds = development_folds()
    assert [(fold.train, fold.validation) for fold in folds] == [
        ((1, 19), (20, 24)),
        ((1, 24), (25, 29)),
        ((1, 29), (30, 34)),
        ((1, 34), (35, 39)),
    ]
    for plain, embargoed in zip(folds, development_folds(2), strict=True):
        assert embargoed.train == (1, plain.train[1] - 2)
        assert embargoed.validation == plain.validation
        assert embargoed.inner_early_stopping == (
            plain.train[1] - 4,
            plain.train[1] - 2,
        )
    with pytest.raises(ValueError, match="embargo|Embargo"):
        development_folds(-1)
    with pytest.raises(ValueError, match="inner training split"):
        development_folds(19)


@pytest.mark.parametrize("periods", [tuple(range(1, 40))], indirect=True)
def test_walk_forward_fits_only_development_and_deduplicates_runs(
    periods: EllipticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    development = build_window(periods, 1, 39)
    contaminated = development.steps.copy()
    contaminated.iloc[-1] = 40
    with pytest.raises(DatasetError, match="limited to steps"):
        cache_development(
            replace(development, steps=contaminated), {"snapshot_sha256": "a" * 64}
        )
    with pytest.raises(DatasetError, match="outside"):
        slice_development(development, (38, 40))
    cache = cache_development(development, {"snapshot_sha256": "a" * 64})
    output = tmp_path / "experiments"
    result = run_baseline_folds(cache, output, git_commit="baseline-test")
    summary = read_json(result)
    assert summary["embargo"] == 0
    assert len(summary["folds"]) == 8
    assert {row["model"] for row in summary["folds"]} == {
        "logistic_regression",
        "xgboost",
    }
    for row in summary["folds"]:
        fold = next(fold for fold in development_folds() if fold.name == row["fold"])
        report = read_json(output / row["config_sha256"] / "evaluation.json")
        assert report["train"]["last_step"] == fold.train[1]
        assert report["validation"]["support"]["last_step"] == fold.validation[1]
        assert report["validation"]["support"]["unknown"] == 5
        assert report["git_commit"] == "baseline-test"
        assert (
            report["config"]["implementation_sha256"] == report["implementation_sha256"]
        )
        assert report["config"]["environment"] == report["environment"]
        assert {"numpy", "pandas", "scikit-learn", "xgboost"} <= set(
            report["environment"]["packages"]
        )

    def unexpected_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("A duplicate experiment should reuse its saved fit report.")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    assert run_baseline_folds(cache, output, git_commit="later-revision") == result
    from mlflow import MlflowClient

    client = MlflowClient(
        tracking_uri=f"sqlite:///{(output / 'tracking/mlflow.db').as_posix()}"
    )
    experiment = client.get_experiment_by_name(TRACKING_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert {run.info.run_id for run in runs} == {
        row["mlflow_run_id"] for row in summary["folds"]
    }
    assert all(run.info.status == "FINISHED" for run in runs)


@pytest.mark.parametrize("change", ["implementation", "numpy"])
@pytest.mark.parametrize("periods", [(1, 19, 20, 24)], indirect=True)
def test_fold_report_and_mlflow_cache_follow_implementation_and_dependencies(
    change: str,
    periods: EllipticDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        experiments, "development_folds", lambda embargo: development_folds(embargo)[:1]
    )
    candidates = experiments._candidates
    monkeypatch.setattr(
        experiments,
        "_candidates",
        lambda: {"logistic_regression": candidates()["logistic_regression"]},
    )
    cache = cache_development(
        build_window(periods, 1, 39), {"snapshot_sha256": "b" * 64}
    )
    output = tmp_path / "experiments"
    before = run_baseline_folds(cache, output, git_commit="old-revision")
    if change == "implementation":
        monkeypatch.setattr(experiments, "implementation_digest", lambda: "d" * 64)
    else:
        environment = experiments.evaluation_environment()
        environment["packages"]["numpy"] = "changed-version"
        monkeypatch.setattr(experiments, "evaluation_environment", lambda: environment)
    after = run_baseline_folds(cache, output, git_commit="new-revision")

    (old_fold,) = read_json(before)["folds"]
    (new_fold,) = read_json(after)["folds"]
    assert old_fold["config_sha256"] != new_fold["config_sha256"]
    assert old_fold["mlflow_run_id"] != new_fold["mlflow_run_id"]
    assert (
        read_json(output / new_fold["config_sha256"] / "evaluation.json")["git_commit"]
        == "new-revision"
    )


def test_development_cache_accepts_a_linked_processed_directory(
    periods: EllipticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "processed target"
    target.mkdir()
    link = tmp_path / "data" / "processed"
    link.parent.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        if os.name != "nt":
            pytest.skip(f"Directory symlinks are unavailable: {error}")
        junction = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if junction.returncode:
            pytest.skip("Directory symlinks and NTFS junctions are unavailable.")
    snapshot = "c" * 64
    cache = cache_development(
        build_window(periods, 1, 39), {"snapshot_sha256": snapshot}
    )
    assert cache.resolve() == development_cache_path(snapshot).resolve()
    assert cache.resolve().is_relative_to(target.resolve())
    assert (cache / "manifest.json").is_file()


@pytest.mark.parametrize("module_name", ["ingestion.py", "schema.py"])
def test_development_cache_rejects_an_old_ingestion_or_schema_fingerprint(
    module_name: str,
    periods: EllipticDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    snapshot = "d" * 64
    old = cache_development(build_window(periods, 1, 39), {"snapshot_sha256": snapshot})
    original_digest = experiments.file_digest

    def changed_digest(path: Path) -> str:
        return "e" * 64 if path.name == module_name else original_digest(path)

    monkeypatch.setattr(experiments, "file_digest", changed_digest)
    assert development_cache_path(snapshot).resolve() != old.resolve()
    with pytest.raises(DatasetError, match="development-only T0 contract"):
        run_baseline_folds(old, tmp_path / "experiments", git_commit="new-revision")


@pytest.mark.parametrize("periods", [tuple(range(1, 40))], indirect=True)
def test_installed_baseline_command_requires_clean_git_and_records_head(
    periods: EllipticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    ignore = tmp_path / ".gitignore"
    rules = "/data/processed/\n/artifacts/\n"
    ignore.write_text(rules, encoding="utf-8")
    git("add", ".gitignore")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Add synthetic workspace rules",
    )
    revision = git("rev-parse", "HEAD")
    snapshot = "f" * 64
    cache_development(build_window(periods, 1, 39), {"snapshot_sha256": snapshot})
    command = shutil.which(
        "elliptic-baseline-folds", path=sysconfig.get_path("scripts")
    )
    assert command is not None
    args = [command, "--snapshot-sha", snapshot]

    ignore.write_text(rules + "# uncommitted change\n", encoding="utf-8")
    dirty = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert dirty.returncode == 1
    assert "Commit or stash working tree changes" in dirty.stderr
    assert not (tmp_path / "artifacts").exists()

    ignore.write_text(rules, encoding="utf-8")
    (tmp_path / "untracked.txt").touch()
    dirty = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert dirty.returncode == 1
    assert "Commit or stash working tree changes" in dirty.stderr
    assert not (tmp_path / "artifacts").exists()
    (tmp_path / "untracked.txt").unlink()

    clean = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert clean.returncode == 0, clean.stderr
    output = tmp_path / "artifacts" / "experiments"
    (summary,) = output.glob("walk-forward-*.json")
    rows = read_json(summary)["folds"]
    assert len(rows) == 8
    assert {
        read_json(output / row["config_sha256"] / "evaluation.json")["git_commit"]
        for row in rows
    } == {revision}
