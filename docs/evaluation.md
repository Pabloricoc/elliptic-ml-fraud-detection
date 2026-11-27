# Temporal evaluation protocol

The prediction unit is a transaction at its scoring time. This protocol fixes the windows, eligible predictors, selection criteria and queue budget for the first modelling experiment.

## Phase 1 handoff

The loader returns `transactions` with `tx_id`, `time_step`, 165 anonymous numeric columns and the original `class`. The separate `edges` table contains `source_id` and `target_id`. Preserve `tx_id` as a join key and `class` as the target. Use an explicit, time-audited list of predictors when building the baseline.

Full-input counts, quantiles and graph summaries are descriptive audit outputs. Make feature-selection, preprocessing and tuning decisions within the appropriate chronological windows. Even neighbours from the same release time step can represent spending that occurred after the prediction instant.

## Phase 2 experiment

The split was fixed by the number of release time steps before comparing models. Steps **1–29** are training, **30–39** are validation, and **40–49** are the final test. Split before deriving predictors. Each window uses only payment edges between its own transactions. An observation with an unknown class can provide historical graph context and enter the review queue. Only explicit illicit and licit labels contribute to fitting and supervised metrics.

The scoring instant is transaction confirmation. Its inputs identify the transactions that created the outputs being spent. The feature allowlist is the count of distinct observed parents, their mean and maximum observed parent counts, and the longest observed path from a root to the transaction. Roots have zero for these four counts or summaries. Parallel edges to the same parent count once. A directed cycle invalidates the ancestry interpretation. These features depend only on incoming ancestry. Later spending leaves earlier feature values unchanged. Counts describe the released subgraph, whose coverage may differ from a live historical feed. [Bitcoin transaction inputs](https://developer.bitcoin.org/devguide/transactions.html)

The release anonymises its transaction attributes and includes aggregates of both upstream and downstream neighbours. Individual columns lack a verified availability time for this use case. Phase 2 excludes all 165 anonymous columns, transaction identifiers, time-step numbers, outgoing edges and labels from predictors. IDs are used for joins and reproducible tie-breaking. [Elliptic paper, section 2.1](https://arxiv.org/pdf/1908.02591)

Compare a standardised logistic regression (`C=1`) with CPU XGBoost (100 trees, depth 3, learning rate 0.08). Use seed 42 and fit both once on the labelled training rows, with unit sample weights. Select the highest illicit-class average precision on validation, favouring logistic regression on an exact tie. For the selected model, maximise validation F1 and take the highest threshold on a tie. Freeze the fitted model and threshold before evaluating the final window. The reference average precision of a constant score is the labelled positive fraction in that window. [Average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

The illustrative review budget is **50 transactions per time step**, fixed before training. At the end of each step, rank all eligible transactions by descending score, breaking ties by a deterministic hash of the seed and transaction ID. Report confirmed illicit hits, reviewed unknowns and recall of known illicit cases across the full queue. Record class support and label coverage alongside the supervised metrics. Ground-truth label publication dates are unavailable, so the experiment assumes earlier-window labels are available at fitting time. A live deployment would need to verify label delays and historical data availability.

## Labels and time

- Preserve all three source labels during ingestion. Use illicit as the positive class and licit as the negative class. Keep `unknown` unlabelled and exclude it from supervised fitting and ground-truth metrics. Report label coverage by period.
- Assign whole time steps to consecutive train, validation and test windows. Record the cutoffs and assert that the last step in each window precedes the first in the next. Keep the last window untouched until the model and scoring rule are fixed.
- Use only features observable when a transaction would be scored. An aggregate over a transaction's neighbours may depend on later spending. Start with features whose availability can be established and document exclusions.
- Fit preprocessing, resampling and model parameters within the training window. Select hyperparameters, a threshold or a review budget using validation. Record the feature list, cutoffs, seed and package versions alongside each run.

## What to measure

Report illicit-class average precision on the labelled test subset, plus precision and recall at the operating point chosen during validation. Include the number of labelled transactions and positives per window so that a change in prevalence is visible.

For a review budget of *k* transactions per time step, rank all eligible transactions and take up to *k*. Report confirmed illicit hits, the count of reviewed transactions with unknown labels, and the proportion of known illicit transactions captured. Mark ratios with an empty denominator as undefined. This keeps the queue realistic even when only some outcomes are observable.

## Predictive evaluation protocol

Train and compare models only on steps **1–39**. Use four expanding folds with the following inclusive boundaries. The optional embargo drops the last two training steps in each fold and preserves the validation range. When a model uses early stopping, reserve the final three steps of the resulting training range as an inner validation window. The fixed legacy configurations use all labelled training rows in each fold and have no early stopping.

| Fold | Training | Validation | Inner early stopping, no embargo |
| --- | --- | --- | --- |
| F1 | 1–19 | 20–24 | 17–19 |
| F2 | 1–24 | 25–29 | 22–24 |
| F3 | 1–29 | 30–34 | 27–29 |
| F4 | 1–34 | 35–39 | 32–34 |

- **Average precision (AP)** is the sum of each increase in recall multiplied by precision at that score threshold. Trapezoidal PR-AUC integrates a linearly interpolated curve. Report scikit-learn AP. Compute pooled AP over all known labels in a window. Compute step AP separately and report its mean and sample standard deviation over steps with at least one known illicit transaction. Give each fold equal weight in the development summary. ROC-AUC requires both known classes. Brier score is the mean squared error between the predicted probability and the binary outcome on known labels. [Average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- **Q-lab** ranks only explicitly labelled transactions within each step. For `K_lab` in `{10, 25, 50}`, precision divides illicit hits by the number reviewed, up to `K_lab`. Recall divides them by all known illicit transactions in that step. Average each ratio over the steps where it is defined. Label coverage affects the population represented by this ranking.
- **Q-all** ranks all eligible transactions within each step. Report confirmed illicit, confirmed licit, unlabelled share and recall of known illicit cases at `K` in `{25, 50, 100}`. The operational capacity is 50 per step, with 25 and 100 as sensitivity checks. Uniform sampling without replacement has expected known illicit hits of `min(K, eligible) × known illicit / eligible` in each nonempty step. Sum these expectations and divide observed known illicit hits by the resulting total. Unknown outcomes prevent measuring overall queue precision.
- **Ties** use the existing deterministic hash order for both rankings. For Q-all, count scores strictly above the cutoff, the full tie group at the cutoff and how many of the tied transactions fit in the queue. A tie spans the cutoff when at least one transaction with the same cutoff score remains outside the queue. Undefined values are stored as JSON `null`.
- **Uncertainty** uses 1,000 whole-step bootstrap draws with seed 42. Resample steps with replacement and recompute the macro mean for AP and Q-lab precision. Report the 2.5th and 97.5th percentiles and the number of draws where the metric is defined. These descriptive intervals treat steps as exchangeable and do not model temporal dependence. Compare candidates using paired differences on shared folds. Report pooled AP on steps 40–42 and 43–49 separately as a descriptive diagnostic when the window spans both.

The development baseline fits fresh models using T0 confirmation-time ancestry and the original logistic regression and XGBoost configurations. The fixed 0.5 threshold in fold reports is descriptive. The frozen legacy model retains its original validation-selected threshold for the re-measurement. The API scenario (S1) is restricted to ancestry tiers T0/T1. A1 applies only to end-of-step batch triage (S2), whose future results will be labelled conditional on A1.

See the [versioned feature and inference contracts](feature-sets.md) for the tiers, source column boundaries, and batch explanation units.

Phases 3 and 4 read only the verified development export at `data/processed/development/`. The [phase 3 ablation report](experiments/phase3-ablation.md) records the fold comparisons, tier decisions and A1 sensitivity. The [test ledger](experiments/test-ledger.md) records the authorised export access.

The development cache contains steps 1–39 only. Its key includes the dataset fingerprint, T0 version and the code for ingestion, validation, schema and feature construction. From a clean Git checkout, run `elliptic-baseline-folds --snapshot-sha <sha256>` against an existing compatible cache. Add `--embargo 2` for the sensitivity check. The command records the Git revision and reuses fold reports and MLflow runs only when the experiment configuration, implementation digest and numerical environment match.

Fold reports are written under `artifacts/experiments/<config-hash>/` and logged to a separate local MLflow store under `artifacts/experiments/tracking/`. An authorised holdout replay publishes its aggregate-only `evaluation.json` in a new directory under `artifacts/evaluations/` before attempting optional cache creation. If the cache write fails, the published report remains available. Record each authorised holdout access in the [test access ledger](experiments/test-ledger.md). Follow the [pre-registration template](experiments/preregistration.md) before final evaluation.

## Recorded local result

The 25 November 2026 run used the full local Elliptic release, seed 42, Python 3.12.14, scikit-learn 1.7.2 and XGBoost 3.1.2. The upstream Kaggle version number was unavailable. The manifest identifies the input snapshot by SHA-256 (`903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043`) and records the hash of each original CSV.

| Window | Transactions | Labelled | Known illicit | Label coverage |
| --- | ---: | ---: | ---: | ---: |
| Train, 1–29 | 120,804 | 26,381 | 2,871 | 21.84% |
| Validation, 30–39 | 36,318 | 8,999 | 1,038 | 24.78% |
| Test, 40–49 | 46,647 | 11,184 | 636 | 23.98% |

Validation AP was **0.1370** for logistic regression and **0.2025** for XGBoost. XGBoost advanced to the final test with a validation-selected score threshold of **0.168804** (rounded). On the labelled test subset its AP was **0.0882**. A constant score on the same subset gives **0.0569**. Applying the frozen threshold yielded 328 true positives, 3,227 false positives, 308 false negatives and 7,321 true negatives. Precision was **9.23%**, recall **51.57%** and F1 **0.1565**.

The separate end-of-step ranking selected 500 of the 46,647 eligible transactions. These 500 contained **5 known illicit**, **39 known licit** and **456 unlabelled** transactions. The queue captured **0.79% of the 636 known illicit cases**. The 456 unknown outcomes prevent measuring the queue's overall precision. Future comparisons should continue tuning within the development windows and disclose any reuse of this test period.

### Legacy re-measurement, 27 November 2026

The [ledger](experiments/test-ledger.md) records a single re-measurement of the frozen T0 XGBoost on steps 40–49. The report reproduced pooled AP **0.08823516261360626**, the operating point (**328 TP, 3227 FP, 308 FN, 7321 TN**) and the number of confirmed illicit cases in the top-50 queue. The model's scores also matched the saved batch replay exactly.

Mean step AP was **0.116532** (sample SD 0.093125, 95% bootstrap interval 0.062393–0.171766). ROC-AUC was **0.625770** and Brier score was **0.056860**. Pooled AP was 0.199342 for steps 40–42 and 0.028190 for steps 43–49. The split is descriptive.

| Q-lab capacity | Mean step precision | Mean step recall | 95% bootstrap interval for precision |
| ---: | ---: | ---: | ---: |
| 10 | 0.100000 | 0.063477 | 0.040000–0.160000 |
| 25 | 0.096000 | 0.073420 | 0.024000–0.176000 |
| 50 | 0.096000 | 0.096397 | 0.019950–0.180000 |

| Q-all capacity | Reviewed | Confirmed illicit | Confirmed licit | Unlabelled | Recall of known illicit | Random expected hits | Observed / expected | Steps with ties spanning the cutoff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 250 | 2 | 22 | 226 (90.4%) | 0.003145 | 3.262781 | 0.612974 | 10 |
| 50 | 500 | 5 | 39 | 456 (91.2%) | 0.007862 | 6.525561 | 0.766218 | 10 |
| 100 | 1000 | 14 | 97 | 889 (88.9%) | 0.022013 | 13.051122 | 1.072705 | 10 |

At the operational capacity of 50 per step, the queue recovered **5 of 636** known illicit cases. The cutoff tie spanned the queue boundary in every step, with 7 steps containing a cutoff group of more than 500 transactions. The deterministic hash order decided which tied cases filled the remaining slots in those groups.

| Step | AP | Strictly above the top-50 cutoff | Tied at cutoff | Tied and reviewed |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 0.160165 | 47 | 16 | 3 |
| 41 | 0.239635 | 1 | 1149 | 49 |
| 42 | 0.203043 | 4 | 1484 | 46 |
| 43 | 0.018380 | 40 | 693 | 10 |
| 44 | 0.054823 | 27 | 24 | 23 |
| 45 | 0.003738 | 7 | 961 | 43 |
| 46 | 0.251603 | 2 | 804 | 48 |
| 47 | 0.026254 | 41 | 1173 | 9 |
| 48 | 0.086124 | 12 | 532 | 38 |
| 49 | 0.121554 | 33 | 44 | 17 |

## Walk-forward reference on development data

Both T0 configurations were fit independently in each of the four folds with no embargo. All eight fits used the development-only cache and completed in local MLflow. The optional embargo is covered by synthetic tests. These measurements establish a reference on steps 1–39 for future comparisons.

| Fold | Logistic regression, mean step AP | XGBoost, mean step AP |
| --- | ---: | ---: |
| F1 | 0.141449 | 0.199577 |
| F2 | 0.234714 | 0.249751 |
| F3 | 0.162250 | 0.250149 |
| F4 | 0.168463 | 0.169949 |
| Mean over folds | 0.176719 | 0.217357 |
| Sample SD over folds | 0.040352 | 0.039532 |
