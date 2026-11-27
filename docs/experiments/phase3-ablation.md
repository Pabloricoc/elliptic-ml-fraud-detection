# Phase 3 feature-tier ablation

Development frame SHA-256: `680aa81f91bfb708de8d8ea01f5eb260e723a0cd59cb7b71cd304526b90c6ca8`.
Original CSV snapshot SHA-256: `903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043`.
Experiment implementation revision: `d9d261023e1055a1d0d4cca70de4bb17c64a2529`.
Python: 3.12.14. Packages: numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.16.3, threadpoolctl 3.6.0, joblib 1.5.3, xgboost 3.1.2.

The experiment uses the four expanding development folds and the phase 3 configuration in `configs/phase3-ablation.json`. XGBoost uses the legacy configuration, including its single-thread setting. Random forest fits 500 trees with minimum leaf size 2 and six workers, then scores with one worker to fix floating-point accumulation order. Both use seed 42, a descriptive threshold of 0.5, and no early stopping or embargo. Each fold fits only known labels. The initial references are T0 XGBoost for S1 and T1b XGBoost for S2. S1 uses only ancestry. **All S2 results are conditional on A1.**

From a clean checkout with the existing verified export in `data/processed/development/`, run `elliptic-experiment --config configs/phase3-ablation.json`. The ablation reads only the export.

AP is averaged over validation steps with known illicit cases and then over folds. The table shows sample SD across folds and the paired AP difference against the declared tier with the same family. Labelled P@50 is averaged over steps where it is defined. Queue lift is summed known hits divided by summed uniform random expectations at 50 slots per step. Tie mass is the number of transactions tied at the cutoff.

| Scenario | Feature set | Family | Mean step AP ± SD | Paired tier reference | Δ AP ± SE | Labelled P@50 | Queue lift | Mean cutoff ties | Steps with ≤5 cutoff ties |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| S1 | T0 | xgboost | 0.217357 ± 0.039532 | — | — | 0.241000 | 1.898977 | 218.60 | 0/20 |
| S1 | T0 | random_forest | 0.212047 ± 0.043572 | — | — | 0.230000 | 1.606827 | 46.20 | 9/20 |
| S1 | T1 | xgboost | 0.208453 ± 0.052233 | T0 | -0.008904 ± 0.008614 | 0.233000 | 1.431537 | 162.30 | 0/20 |
| S1 | T1 | random_forest | 0.201188 ± 0.054885 | T0 | -0.010858 ± 0.007383 | 0.202000 | 1.402322 | 18.45 | 10/20 |
| S2 | T1b | xgboost | 0.216729 ± 0.032730 | — | — | 0.193000 | 1.694472 | 152.20 | 0/20 |
| S2 | T1b | random_forest | 0.213919 ± 0.050294 | — | — | 0.195000 | 1.811332 | 57.15 | 11/20 |
| S2 | T2 | xgboost | 0.909773 ± 0.082159 | T1b | 0.693043 ± 0.039451 | 0.857000 | 16.039053 | 13.35 | 10/20 |
| S2 | T2 | random_forest | 0.919372 ± 0.087392 | T1b | 0.705453 ± 0.041989 | 0.857000 | 18.200966 | 27.95 | 10/20 |
| S2 | T1b+T2 | xgboost | 0.911652 ± 0.082030 | T1b | 0.694923 ± 0.039732 | 0.856000 | 16.623354 | 10.70 | 14/20 |
| S2 | T1b+T2 | random_forest | 0.919302 ± 0.087837 | T1b | 0.705383 ± 0.042386 | 0.858000 | 17.558235 | 29.10 | 11/20 |
| S2 | T2agg+T2 | xgboost | 0.913187 ± 0.080662 | T2 | 0.003415 ± 0.001071 | 0.858000 | 16.535709 | 2.20 | 18/20 |
| S2 | T2agg+T2 | random_forest | 0.917153 ± 0.088160 | T2 | -0.002219 ± 0.001026 | 0.859000 | 18.142536 | 1.85 | 19/20 |
| S2 | T1b+T2+T3 | xgboost | 0.918864 ± 0.075845 | T1b+T2 | 0.007212 ± 0.003220 | 0.862000 | 17.207655 | 3.25 | 16/20 |
| S2 | T1b+T2+T3 | random_forest | 0.924275 ± 0.080846 | T1b+T2 | 0.004973 ± 0.003652 | 0.864000 | 18.113321 | 2.85 | 17/20 |
| S2 | T1b+T2+T2agg+T3 | xgboost | 0.921685 ± 0.079051 | T1b+T2+T3 | 0.002821 ± 0.003562 | 0.869000 | 17.178440 | 1.65 | 20/20 |
| S2 | T1b+T2+T2agg+T3 | random_forest | 0.922247 ± 0.087721 | T1b+T2+T3 | -0.002028 ± 0.003922 | 0.863000 | 17.645880 | 1.80 | 19/20 |
| S2 | T1b+T2+T2agg+T3 minus A1 flags | xgboost | 0.919283 ± 0.078789 | T1b+T2+T2agg+T3 | -0.002402 ± 0.000839 | 0.864000 | 17.733525 | 1.60 | 20/20 |

## Original decisions

Arms are considered in configuration order, with XGBoost before random forest. A candidate must improve AP in at least three folds, exceed one paired SE on average, match the incumbent's labelled P@50 and have at most five tied transactions in at least 90% of steps. Within one paired SE, prefer fewer columns and then fewer trees when both operational constraints pass.

- **S1** retains T0 (xgboost), which remains a reference because the cutoff-tie constraint was unmet.
- **S2** retains T1b+T2+T2agg+T3 (xgboost), which meets the cutoff-tie constraint.

| Scenario | Candidate | Incumbent at comparison | Positive folds | Δ AP ± SE | Precision floor | Tie constraint | Outcome |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| S1 | T0 (random_forest) | T0 (xgboost) | 1/4 | -0.005310 ± 0.003642 | fail | fail | retained incumbent |
| S1 | T1 (xgboost) | T0 (xgboost) | 1/4 | -0.008904 ± 0.008614 | fail | fail | retained incumbent |
| S1 | T1 (random_forest) | T0 (xgboost) | 1/4 | -0.016168 ± 0.010370 | fail | fail | retained incumbent |
| S2 | T1b (random_forest) | T1b (xgboost) | 2/4 | -0.002810 ± 0.009672 | pass | fail | retained incumbent |
| S2 | T2 (xgboost) | T1b (xgboost) | 4/4 | 0.693043 ± 0.039451 | pass | fail | retained incumbent |
| S2 | T2 (random_forest) | T1b (xgboost) | 4/4 | 0.702642 ± 0.042582 | pass | fail | retained incumbent |
| S2 | T1b+T2 (xgboost) | T1b (xgboost) | 4/4 | 0.694923 ± 0.039732 | pass | fail | retained incumbent |
| S2 | T1b+T2 (random_forest) | T1b (xgboost) | 4/4 | 0.702572 ± 0.042906 | pass | fail | retained incumbent |
| S2 | T2agg+T2 (xgboost) | T1b (xgboost) | 4/4 | 0.696458 ± 0.038896 | pass | pass | improves |
| S2 | T2agg+T2 (random_forest) | T2agg+T2 (xgboost) | 3/4 | 0.003965 ± 0.004842 | pass | pass | retained incumbent |
| S2 | T1b+T2+T3 (xgboost) | T2agg+T2 (xgboost) | 4/4 | 0.005677 ± 0.002716 | pass | fail | retained incumbent |
| S2 | T1b+T2+T3 (random_forest) | T2agg+T2 (xgboost) | 4/4 | 0.011087 ± 0.002684 | pass | fail | retained incumbent |
| S2 | T1b+T2+T2agg+T3 (xgboost) | T2agg+T2 (xgboost) | 4/4 | 0.008498 ± 0.002163 | pass | pass | improves |
| S2 | T1b+T2+T2agg+T3 (random_forest) | T1b+T2+T2agg+T3 (xgboost) | 3/4 | 0.000562 ± 0.004929 | fail | pass | retained incumbent |

## A1 sensitivity

Within each fold, partial Spearman is computed on training rows for each T2/T3 column against out-degree, controlling for in-degree. The top 5% of defined absolute coefficients are flagged, rounded up. The strongest S2 arm under the decision rule is repeated when it contains flagged inputs. When the structural reference survives, the highest-AP S2 arm with applicable flags is used for this diagnostic. Removing a flagged T2 column also removes its matching parent and grandparent aggregates. The probe cannot establish feature availability, and this diagnostic does not change the tier decision.

| Fold | Training steps | Defined columns | Flagged | Removed predictors |
| --- | --- | ---: | ---: | ---: |
| F1 | 1–19 | 165 | 9 | 37 |
| F2 | 1–24 | 165 | 9 | 37 |
| F3 | 1–29 | 165 | 9 | 37 |
| F4 | 1–34 | 165 | 9 | 41 |

The filtered row repeats **T1b+T2+T2agg+T3 (xgboost)**.

Aggregate artifact: `artifacts/experiments/phase3-eab31c77964ce9c88e68f29c7df8653ba146c71f0bd4b514fb02b8cbcdd36813.json`.

## Decision rule review

The original decisions above required an absolute cutoff-tie pass for every challenger. The initial reference could remain even if it failed that same requirement. This asymmetry rejected candidates that improved AP and reduced ties. The review identified the issue using the saved development fold aggregates and introduced no holdout access.

The corrected comparisons require AP gains in at least three folds and a mean gain greater than one paired SE. A challenger must also meet the incumbent's labelled P@50 and have at most the incumbent's mean cutoff tie mass. Within one paired SE, fewer columns are preferred subject to the precision and relative tie constraints. The absolute cutoff requirement (at most five ties in at least 90% of steps) applies at final pre-registration or deployment.

The corrected tier decisions use the reference paths declared in the configuration, with the same family on each tier edge. This compares a tier with the inputs it replaces or extends. Among arms that pass these comparisons, the highest mean AP is the reference for the paired simplicity check. The running-incumbent replay is reported separately because comparing across feature branches can make the result depend on declaration order.

| Scenario | Candidate | Declared reference | Columns | Positive folds | Δ AP ± SE | Precision floor | Mean ties | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| S1 | T0 (random_forest) | T0 (xgboost) | 4 | 1/4 | -0.005310 ± 0.003642 | fail | pass | retained incumbent |
| S1 | T1 (xgboost) | T0 (xgboost) | 7 | 1/4 | -0.008904 ± 0.008614 | fail | pass | retained incumbent |
| S1 | T1 (random_forest) | T0 (random_forest) | 7 | 1/4 | -0.010858 ± 0.007383 | fail | pass | retained incumbent |
| S2 | T1b (random_forest) | T1b (xgboost) | 4 | 2/4 | -0.002810 ± 0.009672 | pass | pass | retained incumbent |
| S2 | T2 (xgboost) | T1b (xgboost) | 93 | 4/4 | 0.693043 ± 0.039451 | pass | pass | improves |
| S2 | T2 (random_forest) | T1b (random_forest) | 93 | 4/4 | 0.705453 ± 0.041989 | pass | pass | improves |
| S2 | T1b+T2 (xgboost) | T1b (xgboost) | 97 | 4/4 | 0.694923 ± 0.039732 | pass | pass | improves |
| S2 | T1b+T2 (random_forest) | T1b (random_forest) | 97 | 4/4 | 0.705383 ± 0.042386 | pass | pass | improves |
| S2 | T2agg+T2 (xgboost) | T2 (xgboost) | 467 | 4/4 | 0.003415 ± 0.001071 | pass | pass | improves |
| S2 | T2agg+T2 (random_forest) | T2 (random_forest) | 467 | 1/4 | -0.002219 ± 0.001026 | pass | pass | retained incumbent |
| S2 | T1b+T2+T3 (xgboost) | T1b+T2 (xgboost) | 169 | 4/4 | 0.007212 ± 0.003220 | pass | pass | improves |
| S2 | T1b+T2+T3 (random_forest) | T1b+T2 (random_forest) | 169 | 3/4 | 0.004973 ± 0.003652 | pass | pass | improves |
| S2 | T1b+T2+T2agg+T3 (xgboost) | T1b+T2+T3 (xgboost) | 543 | 2/4 | 0.002821 ± 0.003562 | pass | pass | retained incumbent |
| S2 | T1b+T2+T2agg+T3 (random_forest) | T1b+T2+T3 (random_forest) | 543 | 1/4 | -0.002028 ± 0.003922 | fail | pass | retained incumbent |

### Phase 4 candidates

- **S1** carries forward `T0`. The corrected choice is T0 (xgboost) with mean step AP 0.217357. Its absolute tie check is 0/20 steps (fail). A single running incumbent under the corrected comparisons would retain T0 (xgboost).
- **S2** carries forward `T1b+T2+T3`, `T1b+T2+T2agg+T3`. The corrected choice is T1b+T2+T3 (random_forest) with mean step AP 0.924275. Its absolute tie check is 17/20 steps (fail). A single running incumbent under the corrected comparisons would retain T1b+T2+T2agg+T3 (xgboost).

The A1 sensitivity above records the original experiment on the 543-column arm. This review reuses the completed folds, including that original sensitivity.
