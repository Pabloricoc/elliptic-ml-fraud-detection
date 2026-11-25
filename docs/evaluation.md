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

## Recorded local result

The 25 November 2026 run used the full local Elliptic release, seed 42, Python 3.12.14, scikit-learn 1.7.2 and XGBoost 3.1.2. The upstream Kaggle version number was unavailable. The manifest identifies the input snapshot by SHA-256 (`903ac5dfcabafb86638bbceddef49530bdad2743c6cfe71a7d1c78bf0da9f043`) and records the hash of each original CSV.

| Window | Transactions | Labelled | Known illicit | Label coverage |
| --- | ---: | ---: | ---: | ---: |
| Train, 1–29 | 120,804 | 26,381 | 2,871 | 21.84% |
| Validation, 30–39 | 36,318 | 8,999 | 1,038 | 24.78% |
| Test, 40–49 | 46,647 | 11,184 | 636 | 23.98% |

Validation AP was **0.1370** for logistic regression and **0.2025** for XGBoost. XGBoost advanced to the final test with a validation-selected score threshold of **0.168804** (rounded). On the labelled test subset its AP was **0.0882**. A constant score on the same subset gives **0.0569**. Applying the frozen threshold yielded 328 true positives, 3,227 false positives, 308 false negatives and 7,321 true negatives. Precision was **9.23%**, recall **51.57%** and F1 **0.1565**.

The separate end-of-step ranking selected 500 of the 46,647 eligible transactions. These 500 contained **5 known illicit**, **39 known licit** and **456 unlabelled** transactions. The queue captured **0.79% of the 636 known illicit cases**. The 456 unknown outcomes prevent measuring the queue's overall precision. Future comparisons should continue tuning within the development windows and disclose any reuse of this test period.
