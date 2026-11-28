# Phase 5 holdout pre-registration

This protocol freezes the shortlist and analysis of the Elliptic Bitcoin release before final fits or access to steps 40–49. Owner approval and merge of this pre-registration are required before either begins. The API serves the frozen legacy model. Both S2 arms are end-of-step batch models and every S2 estimate is conditional on A1.

## Frozen shortlist

| Arm | Feature contract | Identity and role |
| --- | --- | --- |
| S1 legacy XGBoost | T0 v1, four confirmation-time ancestry predictors | Existing deployed run `model-4e919f2711a948c78cd080cb6f0e3417`, model SHA-256 `b66ee19643d20a1af4e7a393d9d9194f163a04869a2997a04d7d7ebfba9e4282`. Reference for the confirmation-time scenario. |
| S2 Random Forest | T1b+T2+T3 v1, 169 predictors available under A1 at the end of the step | Untuned [Phase 3 reference](phase3-ablation.md), with its recorded forest defaults. Reference for the same S2 feature contract. |
| S2 LightGBM, five-seed average | T1b+T2+T3 v1, 169 predictors available under A1 at the end of the step | Primary S2 candidate `c9a07751a87e7cf41593c7943841f16eaf710f47e22fa985e43a6e06a60f88e7` from the [Phase 4 review replay](phase4-tuning.md#review-replay-at-a-2000-round-ceiling). |

The primary candidate passed the [development review](phase4-tuning.md#proposed-phase-5-shortlist) with labelled P@50 of 0.877000 against the RF floor of 0.864000 and at most five cutoff ties in 20 of 20 validation steps. Every shortlisted arm will appear in the final report. Transaction IDs join records and break queue ties. Time steps define feature and evaluation windows. Both fields are excluded from the predictors. Class `1` is illicit, `2` is licit and `unknown` stays unlabelled. Fit and compute supervised metrics using known labels only. All eligible transactions enter Q-all, including unlabelled rows.

## Refit and freeze before access

Use the verified development export for steps 1–39. Its frame SHA-256 is `680aa81f91bfb708de8d8ea01f5eb260e723a0cd59cb7b71cd304526b90c6ca8` and source release snapshot SHA-256 is `903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043`. Compute each feature matrix on its own complete step using the registered predictor order. Record the source revision, feature set version, code and model digests, split, label policy, params, package versions and seeds. The development review used Python 3.12.14, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.16.3, XGBoost 3.1.2 and LightGBM 4.7.0. Reuse that numerical environment for the final run. The nominal folds used train/validation windows of 1–19/20–24, 1–24/25–29, 1–29/30–34 and 1–34/35–39.

- Load the exact legacy S1 artifact read-only. It retains its original training window of steps 1–29 and its deployed threshold.
- Fit the S2 Random Forest once on labelled steps 1–39 with the Phase 3 settings `n_estimators=500`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=6` and the remaining scikit-learn defaults. Its fixed 500 trees require no round selection.
- Fit the S2 LightGBM on labelled steps 1–39 at **339 rounds per seed**, the median of the nominal F1–F4 best rounds `[338, 35, 465, 340]`. Use seeds `[42, 43, 44, 45, 46]` and the unweighted arithmetic mean of their positive-class probabilities. Fit each seed with the same preselected round count and hyperparameters below. Derive the positive weight as the square root of the ratio of known licit to known illicit training rows. Use binary objective, deterministic CPU fitting, `force_col_wise=True`, `subsample_freq=1` and six training threads. Fixed rounds on the final refit use the full development training window.

```json
{
  "num_leaves": 69,
  "min_child_samples": 288,
  "colsample_bytree": 0.30861781869545646,
  "subsample": 0.5142391455699857,
  "reg_lambda": 6.070701514183066,
  "learning_rate": 0.15533612430856186,
  "weight": "sqrt_ratio"
}
```

Freeze each new S2 candidate as a separate version 2 model run in an ignored experiment output directory. The LightGBM run freezes all five members, their order and their probability averaging rule. The legacy reference retains its canonical version 1 digest.

### Frozen diagnostic operating thresholds

The thresholds maximise pooled F1 on known labels in the F4 validation steps 35–39. If several thresholds attain the maximum, choose the highest observed score. Use the unrounded values below and count a score equal to the threshold as positive. The S2 predictions came from models trained on steps 1–34, with LightGBM round selection confined to steps 32–34. The S1 threshold comes from scoring F4 with the original frozen model fitted on steps 1–29. F4 also falls within the original legacy validation period.

| Arm | Fixed F4 threshold | F4 prediction provenance |
| --- | ---: | --- |
| S1 legacy | `0.16880415380001068` | Frozen legacy scores on the development export. The value matches the deployed threshold. |
| S2 Random Forest | `0.5608666666666665` | F4 cached forest replay `79771191e7b1de84daf78217fb4a21cd0325d612ce9520cd81da2b659628ab71`, verified against the Phase 3 F4 reference. |
| S2 LightGBM | `0.800860653329169` | F4 cached five-seed score average `42a50b2edbcc3cffd786ed63b94f7fef462f11420325e6ddcec9afbabca9e588`. |

The threshold comparison is secondary. The review policy is the top 50 eligible transactions per step.

## Primary analysis on steps 40–49

The primary metric is the equally weighted mean of illicit-class **average precision (AP)** per step, calculated with scikit-learn on known labels within each step. AP is undefined in a step without known illicit cases. Report the count of defined steps and the sample standard deviation when at least two APs are defined. Retain undefined values in the per-step report.

Report both signed comparisons, with the S2 LightGBM mean AP first in each subtraction.

1. S2 LightGBM minus S2 Random Forest. The shared features control for the feature contract. The contrast measures the combined modelling gain of family, tuning and seed averaging.
2. S2 LightGBM minus S1 legacy XGBoost. The contrast estimates the predictive difference between the two operating scenarios, including the extra S2 information and their different fitting configurations.

Use paired whole-step bootstrap with **1,000 draws**, NumPy `default_rng(42)` and a nominal two-sided **95% percentile interval** for each pre-specified contrast. Generate a single matrix of draws by sampling ten step indices with replacement for each draw from steps 40 through 49 in ascending index order. Reuse the draw indices for all arms. Take the mean of the within-step AP differences on paired defined steps in each draw. Omit draws with no paired defined value and report the number of retained draws. Calculate interval endpoints at the 2.5th and 97.5th percentiles. Also report each arm's interval for its step-macro AP using the same draws. An interval wholly above zero favours LightGBM for that comparison on this evaluation period.

## Secondary analysis and report

- Report pooled AP, ROC-AUC on known labels when both classes are present, and Brier score on the unchanged positive probabilities. Report labelled support, illicit prevalence and the unlabelled count by step.
- Report Q-lab precision and recall at `K ∈ {10, 25, 50}` by step and the mean over steps with each metric defined. For Q-lab, rank only the known-label transactions and review at most `K` of them per step. Report step-bootstrap intervals for mean P@K_lab using the fixed draws.
- Report Q-all at `K=50` on all eligible rows. Include reviewed count, confirmed illicit and licit counts, unlabelled count and share, recall of known illicit, expected random hits and observed-to-expected ratio. In a step, random expected hits equal `min(50, eligible) × known_illicit / eligible`. Sum expectations over steps and report the ratio as undefined if the denominator is zero. Include the audit's fixed 25 and 100 capacity rows as descriptive supporting diagnostics.
- Sort each queue by decreasing score with ascending SHA-256 of `42|<tx_id>` as the deterministic second key. Use exact scores. For the top-50 cutoff report the number strictly above, the count tied at the cutoff, the number of tied rows reviewed and whether the tie group spans the boundary. Include mean tie mass and the fraction of eligible steps with at most five tied transactions.
- Report pooled AP and mean step AP separately for steps 40–42 and 43–49. This split is descriptive and both comparisons use all defined steps in 40–49.
- At each frozen F4 threshold report the labelled confusion matrix, precision, recall and F1. The thresholds and queue capacity stay fixed through reporting.

Follow the [evaluation protocol](../evaluation.md) for metric definitions. Present an all-arm aggregate table, both paired estimates and intervals, and a complete per-step table with AP, queue outcomes and cutoff ties for each arm. Report the primary and secondary results for all three arms regardless of their ordering. Keep transaction-level scores and raw records in ignored local outputs.

## Single authorised holdout evaluation

After owner approval and merge, record the approval date and merge revision. Refit and verify the S2 artifacts on development, validate the legacy digest, and freeze the model and manifest digests, code revision, numerical environment, feature routing, thresholds and evaluation command before starting the holdout run. Open the release only for the single authorised evaluation and verify it matches the registered source snapshot during that access. Use the same test rows, outcomes and step partitions for all three arms. Attach labels for evaluation after producing scores.

Record the access in the [test ledger](test-ledger.md) with its date, exact code revision, frozen model identities and purpose, including any failed access. Score the three arms in that evaluation, then reuse the resulting scores for all reports, the local batch outputs for S1 and the pre-specified primary S2 arm, and tracking. Publish the aggregate report and record the new runs with the batch and tracking workflows in separate experiment directories. Preserve the existing deployed model and historical artifacts. Keep the API on the frozen legacy S1 model. Any change to the registered protocol or extra holdout access requires owner approval.

## Declared limitations

- **A1.** The public dataset supplies anonymous local and neighbour aggregates. End-of-step availability of all these values remains unverified. Both S2 arms and their comparisons are conditional on A1.
- **Embargo sensitivity.** The matched development check for the 169-column LightGBM reused its saved hyperparameters, five seeds and F1–F4 validation windows. With g=2, the last two steps before each validation window were dropped from training. Paired AP moved from 0.935226 at g=0 to 0.904803 at g=2 (−0.030423 ± 0.021832 paired SE), with a decrease in all four folds. Labelled P@50 moved from 0.877 to 0.869. Both runs met the cutoff tie cap in 20/20 steps. This was a fixed-configuration diagnostic (ceiling 2,000, patience 100), run `9208fca825`. The final fit keeps the nominal 339 rounds.
- **Previous holdout exposure.** The [test ledger](test-ledger.md) records the original legacy evaluation, serving replay, independent audit and authorised re-measurement. It also records the filter-only read that produced the development export. The change from step 43 was already known. Treat this as a pre-specified follow-up on an exposed period. A step-block interval describes uncertainty across these ten sampled blocks, with limited ability to generalise to other periods.
- **Labels and operating point.** Unlabelled rows have unknown outcomes, so Q-all reports recovery of known illicit cases and cannot estimate overall queue precision. The F4 threshold transfers to the development refit without recalibration. Performance at that point will be reported as a secondary result.
