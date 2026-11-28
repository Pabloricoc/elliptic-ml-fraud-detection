# Phase 4 tuned boosters and ensembles

Development frames: `680aa81f91bfb708de8d8ea01f5eb260e723a0cd59cb7b71cd304526b90c6ca8`.
Source snapshot: `903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043`.
Experiment revision: `8c851e81d19485657d1067f08f6e2f6ab0778606`.
Python 3.12.14. numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.16.3, threadpoolctl 3.6.0, joblib 1.5.3, xgboost 3.1.2, lightgbm 4.7.0, optuna 4.9.0.

The experiment reads the verified export of steps 1–39 and the saved Phase 3 reference metrics. All S2 results are **conditional on A1**. S1 uses T0 and a bounded T1 attempt to reduce ties. The API remains on the approved legacy model.

Reproduce from a clean checkout with `elliptic-tune --config configs/phase4-tuning.json --phase3-results artifacts/experiments/phase3-eab31c77964ce9c88e68f29c7df8653ba146c71f0bd4b514fb02b8cbcdd36813.json`. Results resume from local Optuna and fold caches.

The plan allows 4 TPE trials per tier and family, at most 8 per family and scenario, and 300 boosting rounds. TPE uses seed 42. Median pruning uses cumulative mean step AP after each fold once two trials complete and two folds have been observed. The family/scenario search stops after 30 attempts without a cumulative +0.003 gain. The optional pairwise ranking objective was outside this budget.

Each outer fold reserves the final three steps of training to choose the round count by log loss with patience 100. A fresh booster then fits the full outer training window for that count. Class weights use known labels from the rows fitted at each stage. Each best trial is also scored using five seeds (42–46) and the round count chosen by seed 42 in the corresponding fold. The remaining seeds refit only the outer training window.

Optuna attempted **32 trials**, including **26 complete** and **6 pruned**. The scheduled run produced **78 evaluated configurations** and **300 saved fold reports** across trials, seed replicas, blends and embargo. Including the selected-arm embargo diagnostic below gives **84 configurations** and **324 fold reports**. Configuration counts group identical parameters, seeds and stages across folds. A pruned trial may have saved a subset of folds.

## Reference arms

| Scenario | Feature set | Family | Mean step AP | Labelled P@50 | Mean cutoff ties | Steps with ≤5 ties |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S1 | T0 | xgboost | 0.217357 | 0.241000 | 218.60 | 0/20 |
| S2 | T1b+T2+T3 | random_forest | 0.924275 | 0.864000 | 2.85 | 17/20 |

## Walk-forward results

AP is the equally weighted mean across F1–F4 of per-step illicit AP on known labels. The comparison uses the Phase 3 rule with the fixed reference for each scenario. Gains must be positive in at least three folds and exceed one paired SE. Labelled P@50 must meet its floor and mean cutoff ties must be at or below the reference. Within one paired SE, fewer columns are preferred under those same operational constraints. A blend must also pass the rule against its parent. The absolute tie requirement applies at final pre-registration or deployment. S1 candidates must pass it to advance from this phase.

| Scenario | Arm | Step AP ± fold SD | Δ AP ± paired SE | Positive folds | P@50 | Queue lift | Mean ties | Steps with ≤5 ties | Relative decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S1 | T0 · xgboost · TPE (`3836b5c132`) | 0.198428 ± 0.045171 | -0.018928 ± 0.018224 | 1/4 | 0.220000 | 1.080956 | 546.70 | 0/20 | reference retained |
| S1 | T0 · xgboost · TPE (`b8fcc62636`) | 0.200508 ± 0.035978 | -0.016848 ± 0.019074 | 2/4 | 0.222000 | 1.431537 | 745.85 | 0/20 | reference retained |
| S1 | T0 · xgboost · TPE (`d6854745cd`) | 0.195436 ± 0.048335 | -0.021920 ± 0.017991 | 1/4 | 0.217000 | 1.256246 | 758.20 | 0/20 | reference retained |
| S1 | T0 · xgboost · 5 seeds (`db0b04ba74`) | 0.208199 ± 0.034892 | -0.009158 ± 0.014750 | 2/4 | 0.238000 | 1.431537 | 363.80 | 0/20 | reference retained |
| S1 | T0 · lightgbm · TPE (`5a2bbe373e`) | 0.206352 ± 0.034057 | -0.011005 ± 0.004527 | 0/4 | 0.251000 | 1.606827 | 158.75 | 0/20 | reference retained |
| S1 | T0 · lightgbm · TPE (`48e7ccdaf5`) | 0.216769 ± 0.039232 | -0.000588 ± 0.002788 | 2/4 | 0.237000 | 1.519182 | 70.85 | 1/20 | reference retained |
| S1 | T0 · lightgbm · 5 seeds (`03f878e334`) | 0.212453 ± 0.034186 | -0.004903 ± 0.004327 | 1/4 | 0.232000 | 1.519182 | 95.40 | 4/20 | reference retained |
| S1 | T1 · xgboost · TPE (`ab3f481336`) | 0.200375 ± 0.048410 | -0.016981 ± 0.018444 | 1/4 | 0.239000 | 1.489967 | 168.20 | 2/20 | reference retained |
| S1 | T1 · xgboost · TPE (`fe8eb4c853`) | 0.193947 ± 0.040225 | -0.023409 ± 0.008737 | 0/4 | 0.202000 | 1.139386 | 209.20 | 1/20 | reference retained |
| S1 | T1 · xgboost · TPE (`62fdedff93`) | 0.205861 ± 0.040353 | -0.011496 ± 0.007144 | 1/4 | 0.232000 | 1.431537 | 95.80 | 1/20 | reference retained |
| S1 | T1 · xgboost · TPE (`bfc9c5f602`) | 0.208684 ± 0.052988 | -0.008672 ± 0.008896 | 2/4 | 0.230000 | 1.606827 | 228.90 | 2/20 | reference retained |
| S1 | T1 · xgboost · 5 seeds (`591c64ee5c`) | 0.208345 ± 0.054185 | -0.009011 ± 0.009484 | 2/4 | 0.232000 | 1.606827 | 121.80 | 4/20 | reference retained |
| S1 | T1 · lightgbm · TPE (`aab807bbca`) | 0.200653 ± 0.046186 | -0.016703 ± 0.005049 | 0/4 | 0.233000 | 1.694472 | 142.15 | 3/20 | reference retained |
| S1 | T1 · lightgbm · TPE (`064716453d`) | 0.207253 ± 0.058890 | -0.010104 ± 0.010385 | 2/4 | 0.225000 | 1.489967 | 45.55 | 5/20 | reference retained |
| S1 | T1 · lightgbm · TPE (`c2566cb0bc`) | 0.209457 ± 0.062483 | -0.007899 ± 0.011880 | 2/4 | 0.229000 | 1.373107 | 38.05 | 3/20 | reference retained |
| S1 | T1 · lightgbm · 5 seeds (`78397188c1`) | 0.208745 ± 0.059820 | -0.008611 ± 0.010966 | 2/4 | 0.223000 | 1.489967 | 35.45 | 3/20 | reference retained |
| S2 | T1b+T2+T3 · xgboost · TPE (`f41a7fcddf`) | 0.913753 ± 0.073182 | -0.010521 ± 0.008080 | 1/4 | 0.865000 | 17.616665 | 2.00 | 19/20 | reference retained |
| S2 | T1b+T2+T3 · xgboost · TPE (`26e1402398`) | 0.890207 ± 0.069375 | -0.034068 ± 0.020291 | 0/4 | 0.855000 | 17.587450 | 12.00 | 16/20 | reference retained |
| S2 | T1b+T2+T3 · xgboost · TPE (`17e7797a4f`) | 0.937076 ± 0.071405 | 0.012801 ± 0.005230 | 4/4 | 0.877000 | 18.230181 | 7.55 | 18/20 | reference retained |
| S2 | T1b+T2+T3 · xgboost · TPE (`f67fc4fcfd`) | 0.916134 ± 0.076219 | -0.008141 ± 0.005219 | 1/4 | 0.863000 | 16.039053 | 48.60 | 9/20 | reference retained |
| S2 | T1b+T2+T3 · xgboost · 5 seeds (`3765e11f43`) | 0.937179 ± 0.072637 | 0.012904 ± 0.004739 | 4/4 | 0.875000 | 18.639191 | 1.55 | 19/20 | reference retained |
| S2 | T1b+T2+T3 · lightgbm · TPE (`47df2653a6`) | 0.929082 ± 0.078754 | 0.004807 ± 0.002178 | 4/4 | 0.865000 | 17.061580 | 1.05 | 20/20 | passes |
| S2 | T1b+T2+T3 · lightgbm · TPE (`1bca56b300`) | 0.921395 ± 0.077228 | -0.002880 ± 0.006046 | 2/4 | 0.870000 | 16.185129 | 32.70 | 11/20 | reference retained |
| S2 | T1b+T2+T3 · lightgbm · TPE (`0e43521507`) | 0.933038 ± 0.075772 | 0.008763 ± 0.003716 | 3/4 | 0.873000 | 18.931342 | 1.15 | 20/20 | passes |
| S2 | T1b+T2+T3 · lightgbm · 5 seeds (`e0d982c972`) | 0.934870 ± 0.079294 | 0.010596 ± 0.003134 | 4/4 | 0.876000 | 18.931342 | 1.00 | 20/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · xgboost · TPE (`a260486b58`) | 0.920637 ± 0.076589 | -0.003638 ± 0.008448 | 3/4 | 0.870000 | 17.850386 | 1.45 | 19/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · xgboost · TPE (`fda4c83d7d`) | 0.895835 ± 0.072743 | -0.028440 ± 0.020747 | 1/4 | 0.860000 | 17.324515 | 11.85 | 16/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · xgboost · TPE (`7b30fa7154`) | 0.936738 ± 0.072324 | 0.012463 ± 0.004908 | 3/4 | 0.875000 | 18.200966 | 4.70 | 18/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · xgboost · TPE (`a320e571bc`) | 0.923304 ± 0.079581 | -0.000970 ± 0.002682 | 2/4 | 0.863000 | 15.483968 | 56.95 | 8/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · xgboost · 5 seeds (`74c6f2e74c`) | 0.936787 ± 0.071558 | 0.012512 ± 0.005214 | 3/4 | 0.876000 | 18.288611 | 4.05 | 18/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · lightgbm · TPE (`22ab06588c`) | 0.929299 ± 0.078943 | 0.005025 ± 0.003122 | 3/4 | 0.864000 | 16.827859 | 1.10 | 20/20 | passes |
| S2 | T1b+T2+T2agg+T3 · lightgbm · TPE (`be14d6cc05`) | 0.922762 ± 0.077807 | -0.001513 ± 0.004069 | 2/4 | 0.867000 | 16.039053 | 33.70 | 11/20 | reference retained |
| S2 | T1b+T2+T2agg+T3 · lightgbm · TPE (`960813654a`) | 0.934246 ± 0.075003 | 0.009971 ± 0.004502 | 3/4 | 0.879000 | 18.697622 | 1.00 | 20/20 | passes |
| S2 | T1b+T2+T2agg+T3 · lightgbm · 5 seeds (`33afc8a2c7`) | 0.936363 ± 0.074786 | 0.012088 ± 0.004076 | 4/4 | 0.880000 | 18.814482 | 1.00 | 20/20 | passes |
| S2 | T1b+T2+T3 / T1b+T2+T2agg+T3 / T1b+T2+T3 · rank_ensemble · rank average (`87f9346f21`) | 0.936056 ± 0.074424 | 0.011781 ± 0.003932 | 4/4 | 0.877000 | 18.668406 | 1.20 | 20/20 | reference retained |

### Search dispositions and selected rounds

| Scenario | Tier | Family | Trial states in order | Best trial ID | Best rounds by fold |
| --- | --- | --- | --- | --- | --- |
| S1 | T0 | xgboost | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 COMPLETE (4/4), 4 PRUNED (2/4) | `b8fcc62636` | 116, 16, 1, 16 |
| S1 | T0 | lightgbm | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 PRUNED (2/4), 4 PRUNED (2/4) | `48e7ccdaf5` | 10, 2, 11, 11 |
| S1 | T1 | xgboost | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 COMPLETE (4/4), 4 COMPLETE (4/4) | `bfc9c5f602` | 300, 102, 1, 52 |
| S1 | T1 | lightgbm | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 COMPLETE (4/4), 4 PRUNED (2/4) | `c2566cb0bc` | 3, 1, 3, 3 |
| S2 | T1b+T2+T3 | xgboost | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 COMPLETE (4/4), 4 COMPLETE (4/4) | `17e7797a4f` | 300, 94, 298, 299 |
| S2 | T1b+T2+T3 | lightgbm | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 PRUNED (2/4), 4 COMPLETE (4/4) | `0e43521507` | 265, 35, 300, 247 |
| S2 | T1b+T2+T2agg+T3 | xgboost | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 COMPLETE (4/4), 4 COMPLETE (4/4) | `7b30fa7154` | 297, 75, 300, 300 |
| S2 | T1b+T2+T2agg+T3 | lightgbm | 1 COMPLETE (4/4), 2 COMPLETE (4/4), 3 PRUNED (2/4), 4 COMPLETE (4/4) | `960813654a` | 239, 33, 288, 222 |

## Rank averaging

Ranks use average ties and percentiles over all eligible nodes within each step. Pairwise Spearman is averaged over steps with defined correlations. A pair above 0.97 or an undefined average stops the ensemble. If the two boosters pass, the matching Phase 3 forest is replayed on development and checked against its saved fold AP. Rank scores describe a review queue. Their scale is ordinal, so the stored probability-style diagnostics have no calibration interpretation. A within-step rank uses the completed batch and is only a queue diagnostic for S1.

- **S1** stopped at the diversity check. Mean Spearman: xgboost / lightgbm 0.876559, xgboost / random_forest 0.855354, lightgbm / random_forest 0.974028.
- **S2** evaluated. Mean Spearman: xgboost / lightgbm 0.915039, xgboost / random_forest 0.866816, lightgbm / random_forest 0.833699. The paired rule retained the strongest parent. The rank average had an AP delta of −0.001021 ± 0.001513 SE against that parent.

## Two-step embargo sensitivity

This diagnostic repeats the highest-AP configuration in S1 and the two highest-AP configurations in S2 from the per-study bests. A seed average replaces its parent for this ranking only when it meets the relative rule. The diagnostic recomputes inner stopping with shorter training ranges and keeps the same validation steps. The rank ensemble is outside this sensitivity.

| Scenario | Arm | Step AP | Δ vs g=0 ± SE | P@50 | Mean ties | Steps with ≤5 ties |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 | T0 · lightgbm · TPE (`fbad10679e`) | 0.209157 | -0.007612 ± 0.004391 | 0.215000 | 82.20 | 1/20 |
| S2 | T1b+T2+T3 · xgboost · TPE (`c07a29fb9c`) | 0.909216 | -0.027860 ± 0.019972 | 0.872000 | 1.00 | 20/20 |
| S2 | T1b+T2+T2agg+T3 · xgboost · TPE (`45c7877761`) | 0.909300 | -0.027438 ± 0.020283 | 0.873000 | 1.00 | 20/20 |

### Selected S2 candidate under embargo

The AP-based schedule above picked both XGBoost arms in S2. The corrected decision rule selected LightGBM with five seeds. The additional diagnostic used the selected arm's saved hyperparameters (`33afc8a2c7`), the same five seeds and paired validation steps. Seed 42 chose the new round counts on the last three steps of each embargoed training window (285, 139, 224 and 288), then all five seeds refit on the corresponding outer training windows. Optuna trial counts stayed at 32.

| Arm | Step AP | Δ vs g=0 ± SE | Positive folds | Labelled P@50 | Mean ties | Steps with ≤5 ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1b+T2+T2agg+T3 · lightgbm · 5 seeds (`b96aa87897`) | 0.906561 | -0.029802 ± 0.021923 | 0/4 | 0.871000 | 1.00 | 20/20 |

AP fell in all four folds, and both embargoed XGBoost arms ranked above this LightGBM arm on AP. The absolute tie check still passed. This sensitivity qualifies the g=0 decision for review before pre-registration. The local `phase4-8b708dee294cfe64113fc07ae73c89f7a3c6200a0f4fe7ef8da2b0b20dc3aee1-winner-embargo.json` records the 24 additional verified fold reports and their finished MLflow runs.

## Decisions

**S1 negative result.** Among the XGBoost and LightGBM arms, the best cutoff count was 5/20 steps (`064716453d`), below the required 18/20. The highest AP was 0.216769, compared with 0.217357 for the legacy T0 XGBoost. The legacy model remains in the API.

**S2 development candidate conditional on A1.** At g=0, LightGBM with T1b+T2+T2agg+T3 and five seeds (`33afc8a2c7`) scored 0.936363 AP, with a gain of 0.012088 ± 0.004076 SE over the default random forest. It exceeded its single-seed parent by 0.002117 ± 0.000700 SE and met the absolute tie check in 20/20 steps. The embargo result above documents the drop in AP under a longer information gap.

The full local aggregate retains trial parameters, per-fold rounds, paired gate outcomes and fold run digests.
