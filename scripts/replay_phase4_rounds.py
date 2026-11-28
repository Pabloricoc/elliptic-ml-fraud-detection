"""Reevaluate the frozen Phase 4 review arms with a larger boosting cap."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from elliptic_fraud.development import DEVELOPMENT_DIR, load_development_export
from elliptic_fraud.evaluate import evaluation_environment
from elliptic_fraud.experiments import clean_git_revision
from elliptic_fraud.folds import development_folds
from elliptic_fraud.runs import (
    file_digest,
    implementation_digest,
    read_json,
    snapshot_digest,
    write_json,
)
from elliptic_fraud.schema import DatasetError
from elliptic_fraud.tuned_models import PATIENCE
from elliptic_fraud.tuning import _counts, _measured
from elliptic_fraud.tuning_runs import BoosterArm, TuningRun
from elliptic_fraud.tuning_search import Search, score_booster, seed_average

CEILING = 2000
FROZEN_PAIRS = (
    ("0e43521507", "e0d982c972"),
    ("17e7797a4f", "3765e11f43"),
    ("960813654a", "33afc8a2c7"),
)


def _frozen_pair(archive: dict, output: Path, prefixes: tuple[str, str]) -> tuple:
    rows = [
        next(arm for arm in archive["arms"] if arm["id"].startswith(prefix))
        for prefix in prefixes
    ]
    base, average = rows
    folds = development_folds()
    if (
        base["scenario"] != "S2"
        or average["scenario"] != "S2"
        or base["stage"] != "TPE"
        or average["stage"] != "5 seeds"
        or any(
            base[key] != average[key]
            for key in ("family", "features", "search_parameters")
        )
    ):
        raise DatasetError("The archived booster and seed average must agree.")
    for row in rows:
        digests = [fold["config_sha256"] for fold in row["folds"]]
        if snapshot_digest({"folds": digests}) != row["id"]:
            raise DatasetError("An archived arm does not match its fold identities.")
        for fold, digest in zip(folds, digests, strict=True):
            report = read_json(output / digest / "evaluation.json")
            config = report["config"]
            if (
                snapshot_digest(config) != digest
                or snapshot_digest(config["fold"]) != snapshot_digest(asdict(fold))
                or {
                    key: value
                    for key, value in config["params"].items()
                    if key != "n_estimators"
                }
                != row["search_parameters"]
            ):
                raise DatasetError("An archived fold does not match the frozen replay.")
    return base, average


def _replay(run: TuningRun, inputs: tuple[dict, dict], reference: dict) -> list[dict]:
    base, average = inputs
    tier, family, params = base["features"], base["family"], base["search_parameters"]
    search = Search("S2", tier, family, 0, CEILING)
    window = run.window("S2", tier)
    folds = development_folds()
    results = [score_booster(run, window, search, fold, params) for fold in folds]
    parent = BoosterArm("S2", tier, family, "fixed", params, results)
    print(
        f"{tier} {family} selected rounds: {parent.aggregate()['fold_rounds']}",
        flush=True,
    )
    seeded = seed_average(run, window, search, parent, folds)
    rows = [
        {**_measured(candidate, reference, comparison), "source": source["id"]}
        for candidate, comparison, source in (
            (parent, None, base),
            (seeded, parent, average),
        )
    ]
    for row in rows:
        # With one inner log-loss metric, training stops after PATIENCE
        # rounds past its last improvement or when it reaches the limit.
        row["inner_iterations"] = [
            min(CEILING, rounds + PATIENCE) for rounds in row["fold_rounds"]
        ]
        row["inner_at_ceiling"] = row["inner_iterations"].count(CEILING)
        row["best_at_ceiling"] = row["fold_rounds"].count(CEILING)
    print(f"{tier} {family} completed four folds and five seeds", flush=True)
    return rows


def main() -> None:
    """Replay the fixed review pairs with no Optuna study access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original local Phase 4 JSON")
    archive_path = parser.parse_args().archive.resolve()
    root = Path(__file__).resolve()
    project = Path.cwd().resolve()
    output = (project / "artifacts" / "experiments").resolve()
    if archive_path.parent != output or archive_path.suffix != ".json":
        raise DatasetError("Use the original local Phase 4 experiment aggregate.")
    git_commit = clean_git_revision()
    archive = read_json(archive_path)
    environment = evaluation_environment()
    environment["packages"].update(
        {name: version(name) for name in ("lightgbm", "optuna")}
    )
    implementation = implementation_digest()
    if (
        archive["environment"] != environment
        or archive["implementation_sha256"] != implementation
    ):
        raise DatasetError("Use the original Phase 4 code and numerical environment.")
    frozen = [_frozen_pair(archive, output, ids) for ids in FROZEN_PAIRS]
    development_dir = (project / DEVELOPMENT_DIR).resolve()
    dataset, manifest = load_development_export(development_dir)
    fingerprints = {
        key: manifest[key]
        for key in ("development_snapshot_sha256", "source_snapshot_sha256")
    }
    if any(archive[key] != value for key, value in fingerprints.items()):
        raise DatasetError("The replay requires the original development snapshot.")
    protocol = {
        "archive_sha256": file_digest(archive_path),
        "script_sha256": file_digest(root),
        "ceiling": CEILING,
        "patience": PATIENCE,
        "embargo": 0,
        "frozen_arms": [[base["id"], average["id"]] for base, average in frozen],
    }
    provenance = {
        **fingerprints,
        "implementation_sha256": implementation,
        "environment": environment,
        "protocol_sha256": snapshot_digest(protocol),
    }
    run = TuningRun(
        dataset,
        development_dir.parent / "phase4-features",
        output,
        provenance,
        git_commit,
    )
    reference = archive["anchors"]["S2"]
    path = output / f"{archive_path.stem}-ceiling-{CEILING}.json"
    with FileLock(path.with_suffix(".lock")):
        rows = [row for pair in frozen for row in _replay(run, pair, reference)]
        summary = {
            **provenance,
            **protocol,
            "git_commit": git_commit,
            "archive": archive_path.name,
            "s2_reference": reference,
            "no_new_trials": True,
            "arms": rows,
            "counts": _counts(run),
        }
        temporary = path.with_suffix(".partial")
        write_json(temporary, summary)
        temporary.replace(path)
    print(f"Fixed-capacity replay written to {path}", flush=True)


if __name__ == "__main__":
    main()
