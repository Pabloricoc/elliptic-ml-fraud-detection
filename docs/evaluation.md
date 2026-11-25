# Temporal evaluation protocol

This is the design for the first modelling experiment. The prediction unit is a transaction at its scoring time. Keep its identifier for joins and auditing, and its time step for partitioning.

## Phase 1 handoff

The loader returns `transactions` with `tx_id`, `time_step`, 165 anonymous numeric columns and the original `class`. The separate `edges` table contains `source_id` and `target_id`. Preserve `tx_id` as a join key and `class` as the target. Use an explicit, time-audited list of predictors when building the baseline.

Full-input counts, quantiles and graph summaries are descriptive audit outputs. Make feature-selection, preprocessing and tuning decisions within the appropriate chronological windows. Even neighbours from the same release time step can represent spending that occurred after the prediction instant.

## Labels and time

- Preserve all three source labels during ingestion. Use illicit as the positive class and licit as the negative class. Keep `unknown` unlabelled and exclude it from supervised fitting and ground-truth metrics. Report label coverage by period.
- Assign whole time steps to consecutive train, validation and test windows. Record the cutoffs and assert that the last step in each window precedes the first in the next. Keep the last window untouched until the model and scoring rule are fixed.
- Use only features observable when a transaction would be scored. An aggregate over a transaction's neighbours may depend on later spending. Start with features whose availability can be established and document exclusions.
- Fit preprocessing, resampling and model parameters within the training window. Select hyperparameters, a threshold or a review budget using validation. Record the feature list, cutoffs, seed and package versions alongside each run.

## What to measure

Report illicit-class average precision on the labelled test subset, plus precision and recall at the operating point chosen during validation. Include the number of labelled transactions and positives per window so that a change in prevalence is visible.

For a review budget of *k* transactions per time step, rank all eligible transactions and take up to *k*. Report confirmed illicit hits, the count of reviewed transactions with unknown labels, and the proportion of known illicit transactions captured. Mark ratios with an empty denominator as undefined. This keeps the queue realistic even when only some outcomes are observable.
