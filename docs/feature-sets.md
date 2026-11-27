# Feature sets and frozen model contracts

Each feature set declares its name, version, scenario, ordered columns, and availability. The builder processes each time step independently and preserves input row order. Transaction IDs are join keys, time steps are window boundaries, and labels are outcomes. All three are excluded from predictors.

| Tier | Scenario | Predictors and timing |
| --- | --- | --- |
| T0 | S1, S2 | The four legacy ancestry features, with their original order and values. |
| T1 | S1, S2 | Distinct ancestors, distinct ancestors reachable by a path of exactly two edges, maximum funding depth, mean/max/population SD of ancestor in-degree, and distinct ancestral roots. Uses incoming funding history. |
| T1b | S2 | Distinct direct recipients, distinct descendants, weakly connected component size and PageRank using the completed step. PageRank uses damping 0.85 and a uniform distribution for dangling nodes. |
| T2 | S2 | The 93 anonymous local columns, `feature_001` through `feature_093`. |
| T2agg | S2 | Mean, maximum and population SD of T2 values over distinct direct parents, and the mean over distinct grandparents reached by two edges. |
| T3 | S2 | The 72 released neighbour aggregates, `feature_094` through `feature_165`. |

Missing ancestor groups have zero-valued statistics. T2agg appends the binary indicators `has_parents` and `has_grandparents` to distinguish absent groups from observed zero-valued aggregates. A composite such as `T1b+T2+T3` takes the ordered union of its tiers, keeping the first occurrence of shared columns. S1 uses confirmation-time ancestry. S2 scores batches at the end of a step. Any S2 result is conditional on A1, the assumption that the anonymous values available in the release use information observable by the end of their step. Ancestry-only traversal in T2agg does not establish when the anonymous inputs became available.

The original paper counts the time-step field in its first 94 features, followed by 72 one-hop backward/forward aggregates. PyTorch Geometric reads the headerless CSV, identifies transaction ID and time step in positions 0 and 1, and preserves the order of the remaining 165 feature columns. Together these sources give 93 local predictors and 72 aggregates once the time step is excluded. [Weber et al., §2.1.2](https://arxiv.org/pdf/1908.02591) · [PyG Elliptic reader](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/elliptic.py)

The versioned joblib cache accepts an explicit absolute root, typically the project's `data/processed/`. Compute the checksum with `development_snapshot_digest(dataset)` on the transactions and edges already restricted to steps 1–39. Before cache lookup, the cache verifies this checksum against the supplied frames, including their values, dtypes and order. Record the full release's raw-file checksum separately as dataset provenance. Cache entries key off the verified development checksum, the ordered feature contract and the implementation SHA-256. A missing or corrupt payload is rebuilt under the cache lock.

## Frozen inference

A manifest without `schema_version` is version 1. The existing training and replay commands and the serving API use the legacy contract. Version 2 batch inference reads `feature_set` with `name`, `version`, `scenario` and `columns`, plus `model_family`, `params` and `fold_protocol`. The loader checks registered column order, model SHA-256, estimator family, fitted feature order and the binary class order. Batch scoring loads only the required anonymous attributes and graph, with outcome labels excluded.

Version 2 accepts logistic regression, XGBoost, LightGBM, random forest and extra trees. LightGBM is available via the `lightgbm` extra and is included in the development environment for tests. Logistic regression explanations require a fitted, mean-centering StandardScaler. The linear SHAP reference is the labelled training mean in the original feature space. Tree families use TreeExplainer with the fitted tree paths. For random forest and extra trees the trees' native positive-class output is a probability. Add `base_probability` to the signed contributions to reconstruct it. For logistic regression, XGBoost and LightGBM, add `base_log_odds` to the contributions and apply the logistic function. The batch report records these units and checks the reconstruction against the scored probabilities. [TreeExplainer output conventions](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
