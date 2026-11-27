"""Cache and track development-only tuning folds and their private scores."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .ablation_report import summarize_arm
from .evaluate import evaluate_window
from .experiments import DEVELOPMENT_THRESHOLD, record_fold_mlflow
from .feature_cache import cached_development_features
from .folds import DEV_STEPS, Fold, slice_development
from .ingestion import EllipticDataset
from .model_features import Window, get_feature_set
from .runs import file_digest, read_json, snapshot_digest, write_json
from .schema import DatasetError
from .train import SEED

Fit = Callable[[Window, Window], tuple[pd.Series, int | None]]


@dataclass(frozen=True)
class FoldResult:
    report: dict
    scores: pd.Series


@dataclass
class TuningRun:
    dataset: EllipticDataset
    cache_root: Path
    output: Path
    provenance: dict
    git_commit: str

    def window(self, scenario: str, tier: str) -> Window:
        """Build or recover the verified predictor window for one feature set."""
        contract = get_feature_set(tier, scenario)
        features, steps = cached_development_features(
            self.dataset,
            self.provenance["development_snapshot_sha256"],
            contract,
            cache_root=self.cache_root,
        )
        labels = self.dataset.transactions.set_index("tx_id")["class"].loc[steps.index]
        return Window(features, labels, steps, *DEV_STEPS)

    def fold(
        self,
        window: Window,
        fold: Fold,
        candidate: dict,
        fit: Fit,
    ) -> FoldResult:
        """Finish a fold once, or recover its verified local scores and aggregates."""
        validation = slice_development(window, fold.validation)
        config = {
            **self.provenance,
            **candidate,
            "schema_version": 2,
            "phase": 4,
            "fold": asdict(fold),
            "inner_early_stopping": fold.inner_early_stopping,
            "embargo": fold.validation[0] - fold.train[1] - 1,
            "review_seed": SEED,
            "diagnostic_threshold": DEVELOPMENT_THRESHOLD,
        }
        digest = snapshot_digest(config)
        folder = self.output / digest
        path = folder / "evaluation.json"
        payload = folder / "development-scores.npy"
        with FileLock(self.output / f".{digest}.lock"):
            if not path.exists():
                train = slice_development(window, fold.train)
                scores, rounds = fit(train, validation)
                if not scores.index.equals(validation.steps.index):
                    raise DatasetError("Tuning scores and validation rows differ.")
                metrics = evaluate_window(
                    validation, scores, seed=SEED, threshold=DEVELOPMENT_THRESHOLD
                )
                folder.mkdir(exist_ok=True)
                temporary_scores = folder / ".scores.partial"
                with temporary_scores.open("wb") as stream:
                    np.save(
                        stream, scores.to_numpy(dtype="float64"), allow_pickle=False
                    )
                temporary_scores.replace(payload)
                report = {
                    "config": config,
                    "config_sha256": digest,
                    "git_commit": self.git_commit,
                    "label_policy": "1 illicit, 2 licit, unknown unlabelled",
                    "availability_assumption": "conditional on A1"
                    if candidate["scenario"] == "S2"
                    else None,
                    "columns": list(window.features.columns),
                    "best_rounds": rounds,
                    "train": train.support(),
                    "validation": metrics,
                    "development_scores_sha256": file_digest(payload),
                }
                temporary = folder / ".evaluation.partial"
                write_json(temporary, report)
                temporary.replace(path)
            report = read_json(path)
            if (
                report.get("config_sha256") != digest
                or snapshot_digest(report["config"]) != digest
                or report["development_scores_sha256"] != file_digest(payload)
            ):
                raise DatasetError(
                    "A stored tuning fold failed its provenance or score digest."
                )
            values = np.load(payload, allow_pickle=False)
            if (
                values.shape != (len(validation.steps),)
                or not np.isfinite(values).all()
            ):
                raise DatasetError(
                    "A stored tuning fold has invalid development scores."
                )
        report["mlflow_run_id"] = record_fold_mlflow(self.output, report)
        return FoldResult(
            report, pd.Series(values, index=validation.steps.index, name="score")
        )


@dataclass
class BoosterArm:
    """One comparable configuration with its saved validation folds."""

    scenario: str
    tier: str
    family: str
    stage: str
    params: dict
    folds: list[FoldResult]

    def aggregate(self) -> dict:
        """Summarize only validation metrics and their local run references."""
        aggregate = summarize_arm(
            self.scenario, self.tier, self.family, [row.report for row in self.folds]
        )
        if "predictor_count" in self.folds[0].report["config"]:
            aggregate["complexity"][0] = self.folds[0].report["config"][
                "predictor_count"
            ]
        return {
            **aggregate,
            "stage": self.stage,
            "search_parameters": self.params,
            "fold_rounds": [row.report["best_rounds"] for row in self.folds],
        }
