"""Freeze the approved Phase 5 estimators on the development export."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .ablation import FAMILY_PARAMS
from .folds import DEV_STEPS, development_folds, slice_development, validate_development
from .inference import FrozenModel, load_frozen_model, predict_scores
from .ingestion import EllipticDataset
from .model_evaluation import validation_threshold
from .model_features import FeatureSet, Window, build_feature_set, get_feature_set
from .runs import file_digest, new_run, read_json, snapshot_digest, write_json
from .schema import DatasetError
from .train import BOOST_PARAMS, SEED
from .tuned_models import SEEDS, booster_parameters

LEGACY_RUN = Path("artifacts/models/model-4e919f2711a948c78cd080cb6f0e3417")
LEGACY_SHA = "b66ee19643d20a1af4e7a393d9d9194f163a04869a2997a04d7d7ebfba9e4282"
TEST_STEPS = (40, 49)
S1_TIER = get_feature_set("T0", "S1")
S2_TIER = get_feature_set("T1b+T2+T3", "S2")
LEGACY_THRESHOLD = 0.16880415380001068
RF_THRESHOLD = 0.5608666666666665
LGBM_THRESHOLD = 0.800860653329169
LGBM_ROUNDS = 339
LGBM_PARAMS = {
    "num_leaves": 69,
    "min_child_samples": 288,
    "colsample_bytree": 0.30861781869545646,
    "subsample": 0.5142391455699857,
    "reg_lambda": 6.070701514183066,
    "learning_rate": 0.15533612430856186,
    "weight": "sqrt_ratio",
}

ARM_NAMES = {
    "s1_legacy": "S1 legacy XGBoost",
    "s1_refit": "S1 T0 XGBoost refit",
    "s2_rf": "S2 Random Forest",
    "s2_lgbm": "S2 LightGBM (5 seeds)",
}


@dataclass(frozen=True)
class Arm:
    """One frozen score function, with one or more ordered model members."""

    members: tuple[FrozenModel, ...]
    feature_set: FeatureSet
    threshold: float

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Predict once per member and take the registered probability mean."""
        scores = (
            predict_scores(member.pipeline, features).astype("float64")
            for member in self.members
        )
        return sum(scores) / len(self.members)


def legacy_model(path: Path) -> FrozenModel:
    """Verify the exact deployed artifact before any evaluation access."""
    manifest = read_json(path / "manifest.json")
    if (
        manifest.get("model_sha256") != LEGACY_SHA
        or manifest.get("threshold") != LEGACY_THRESHOLD
        or manifest.get("winner") != "xgboost"
        or manifest.get("windows_inclusive", {}).get("train") != [1, 29]
    ):
        raise DatasetError("The legacy manifest differs from the registered model.")
    return load_frozen_model(path, manifest)


def _fit(estimator, window: Window) -> Pipeline:
    target = window.known_target()
    if target.nunique() != 2:
        raise DatasetError("The final fit requires both known classes.")
    return Pipeline([("classifier", estimator)]).fit(
        window.features.loc[target.index], target
    )


def _training_window(dataset: EllipticDataset, contract: FeatureSet) -> Window:
    features, steps = build_feature_set(dataset, contract, *DEV_STEPS)
    labels = dataset.transactions.set_index("tx_id")["class"].loc[features.index]
    window = Window(features, labels, steps, *DEV_STEPS)
    validate_development(window)
    return window


def _member(path: Path, model: Pipeline, header: dict, params: dict, seed: int) -> dict:
    path.mkdir(exist_ok=True)
    payload = path / "model.joblib"
    joblib.dump(model, payload, compress=3)
    manifest = {
        **header,
        "params": params,
        "seed": seed,
        "model_sha256": file_digest(payload),
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def freeze_models(
    dataset: EllipticDataset, root: Path, provenance: dict, legacy: FrozenModel
) -> Path:
    """Fit and freeze only the three new arms and the S1 F4 diagnostic."""
    s1 = _training_window(dataset, S1_TIER)
    s2 = _training_window(dataset, S2_TIER)
    f4 = development_folds()[-1]
    f4_train = slice_development(s1, f4.train)
    f4_validation = slice_development(s1, f4.validation)
    f4_model = _fit(XGBClassifier(**BOOST_PARAMS), f4_train)
    f4_scores = predict_scores(f4_model, f4_validation.features)
    threshold = validation_threshold(f4_validation.known_target(), f4_scores)
    identities = {
        "s1_legacy": {
            "name": LEGACY_RUN.name,
            "model_sha256": legacy.sha256,
            "threshold": legacy.threshold,
        }
    }
    fit_plans = (
        ("s1_refit", "xgboost", S1_TIER, s1, threshold, (SEED,)),
        ("s2_rf", "random_forest", S2_TIER, s2, RF_THRESHOLD, (SEED,)),
        ("s2_lgbm", "lightgbm", S2_TIER, s2, LGBM_THRESHOLD, SEEDS),
    )
    with new_run(root, "frozen") as (staging, destination):
        for key, family, contract, window, cut, seeds in fit_plans:
            header = {
                "schema_version": 2,
                "model_family": family,
                "feature_set": contract.manifest(),
                "threshold": cut,
                "fold_protocol": {
                    "final_train": DEV_STEPS,
                    "threshold_source": "F4",
                    "validation": f4.validation,
                },
                "windows_inclusive": {"train": DEV_STEPS, "test": TEST_STEPS},
                "review_budget_per_step": 50,
                **provenance,
            }
            arm_dir = staging / key
            arm_dir.mkdir()
            members = []
            member_manifests = []
            for seed in seeds:
                if family == "lightgbm":
                    params = booster_parameters(
                        family, LGBM_PARAMS, seed, LGBM_ROUNDS, window.known_target()
                    )
                    estimator = LGBMClassifier(**params)
                elif family == "random_forest":
                    params = FAMILY_PARAMS[family]
                    estimator = RandomForestClassifier(**params)
                else:
                    params = BOOST_PARAMS
                    estimator = XGBClassifier(**params)
                model = _fit(estimator, window)
                member_path = arm_dir / f"seed-{seed}" if len(seeds) > 1 else arm_dir
                manifest = _member(member_path, model, header, params, seed)
                members.append(manifest["model_sha256"])
                member_manifests.append(file_digest(member_path / "manifest.json"))
                print(f"Frozen {key} seed {seed}", flush=True)
            if len(seeds) > 1:
                digest = snapshot_digest({"members": members})
                write_json(
                    arm_dir / "manifest.json",
                    {
                        **header,
                        "params": {
                            "tuned": LGBM_PARAMS,
                            "seeds": seeds,
                            "rounds": LGBM_ROUNDS,
                        },
                        "aggregation": "mean_probability",
                        "members": members,
                        "member_manifest_sha256": member_manifests,
                        "model_sha256": digest,
                    },
                )
            else:
                digest = members[0]
            identities[key] = {
                "name": f"{destination.name}/{key}",
                "model_sha256": digest,
                "manifest_sha256": file_digest(arm_dir / "manifest.json"),
                "threshold": cut,
            }
        write_json(
            staging / "freeze.json",
            {
                **provenance,
                "arms": identities,
                "s1_refit_threshold": {
                    "fold": f4.name,
                    "train": f4.train,
                    "validation": f4.validation,
                    "threshold": threshold,
                    "oof_score_sha256": snapshot_digest({"scores": f4_scores.tolist()}),
                },
            },
        )
    return destination


def load_arms(directory: Path, freeze: dict, legacy: FrozenModel) -> dict[str, Arm]:
    """Check frozen identities and deserialize models without inspecting any data."""
    if list(freeze["arms"]) != list(ARM_NAMES):
        raise DatasetError("The freeze lacks the exact registered shortlist.")
    saved_legacy = freeze["arms"]["s1_legacy"]
    if (saved_legacy["model_sha256"], saved_legacy["threshold"]) != (
        legacy.sha256,
        legacy.threshold,
    ):
        raise DatasetError("The legacy identity changed since preparation.")
    arms = {"s1_legacy": Arm((legacy,), S1_TIER, legacy.threshold)}
    thresholds = {
        "s1_refit": freeze["s1_refit_threshold"]["threshold"],
        "s2_rf": RF_THRESHOLD,
        "s2_lgbm": LGBM_THRESHOLD,
    }
    for key, cutoff in thresholds.items():
        path = directory / key
        identity = freeze["arms"][key]
        manifest_path = path / "manifest.json"
        manifest = read_json(manifest_path)
        contract = S1_TIER if key == "s1_refit" else S2_TIER
        if (
            file_digest(manifest_path) != identity["manifest_sha256"]
            or manifest["model_sha256"] != identity["model_sha256"]
            or manifest["threshold"] != cutoff
            or identity["threshold"] != cutoff
            or manifest["feature_set"] != contract.manifest()
        ):
            raise DatasetError(f"Frozen manifest changed for {key}.")
        if key == "s2_lgbm":
            if manifest["aggregation"] != "mean_probability" or manifest["params"] != {
                "tuned": LGBM_PARAMS,
                "seeds": list(SEEDS),
                "rounds": LGBM_ROUNDS,
            }:
                raise DatasetError(
                    "LightGBM members differ from the registered ensemble."
                )
            members = []
            for seed, digest in zip(
                SEEDS, manifest["member_manifest_sha256"], strict=True
            ):
                member = path / f"seed-{seed}"
                member_path = member / "manifest.json"
                if file_digest(member_path) != digest:
                    raise DatasetError("A LightGBM member manifest changed.")
                member_manifest = read_json(member_path)
                frozen = load_frozen_model(member, member_manifest)
                if (
                    member_manifest["seed"] != seed
                    or member_manifest["params"]["random_state"] != seed
                    or member_manifest["params"]["n_estimators"] != LGBM_ROUNDS
                    or frozen.feature_set.manifest() != contract.manifest()
                    or frozen.threshold != cutoff
                ):
                    raise DatasetError(
                        "A frozen LightGBM member has the wrong contract."
                    )
                members.append(frozen)
            hashes = [model.sha256 for model in members]
            if (
                hashes != manifest["members"]
                or snapshot_digest({"members": hashes}) != identity["model_sha256"]
            ):
                raise DatasetError("LightGBM member digests do not match the freeze.")
        else:
            members = [load_frozen_model(path, manifest)]
        arms[key] = Arm(tuple(members), contract, cutoff)
    return arms
