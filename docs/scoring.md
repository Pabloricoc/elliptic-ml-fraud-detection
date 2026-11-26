# Batch scoring and experiment tracking

Phase 3 replays a saved model on complete Elliptic time steps. It supports both Phase 2 winners, including training directories created before the scoring command existed. A run uses the feature order, positive class, validation threshold, ranking seed and review capacity of its training run.

## Inputs and scoring time

Pass a trusted local training directory with its original `model.joblib` and `manifest.json`. The loader verifies the model checksum before deserialising it and checks the saved feature order and class order. A checksum identifies the bytes. Use a model that you trained or whose source you trust.

Point `--data-dir` at the directory containing the original headerless transaction feature CSV and directed edge CSV. Scoring reads the IDs and time steps from the first file and the funding edges from the second. It validates row widths, unique node IDs, edge endpoints and time-step boundaries. Anonymous numeric values are outside the scoring schema. The class CSV is optional and is never opened by the scorer.

The default window is the model's recorded test period. Use `--first-step` and `--last-step` to replay a different inclusive range in the original 49-step release. Supply complete steps so that the observed ancestors of each transaction are available. Scores use the four confirmation-time predictors from training. The queue is formed at the end of each step.

## Read a scoring run

The command prints a new directory under `artifacts/scores/`. Each successful run contains the following outputs.

| File | How to use it |
| --- | --- |
| `scores.csv` | All eligible transaction IDs, time steps, illicit-class scores, ranks and decision flags. A score at or above the frozen threshold sets `above_threshold`. A rank within the per-step review budget sets `in_review_queue`. The score is an uncalibrated model output. |
| `review_queue.csv` | The selected transactions in review order, their four observed feature values, the SHAP base value and one signed SHAP contribution per feature. |
| `summary.json` | Aggregate counts, model and input fingerprints, scoring code and environment, numeric global SHAP values and the maximum score reconstruction error. |
| `shap_importance.png` | Mean absolute SHAP contribution of each feature across **all** scored transactions in this window. |

The decision flags preserve the original comparisons. To re-read exact scores and string IDs in pandas, use `pd.read_csv(path, dtype={"tx_id": "string"}, float_precision="round_trip")`. The phase 2 deterministic tie rule orders equal scores by a hash of the seed and transaction ID.

For XGBoost, TreeSHAP uses the fitted trees' training path weights. For logistic regression, independent linear SHAP uses the labelled training mean as its reference (zero after the fitted scaler). Each local contribution is measured in log-odds. A positive value raises the score relative to the base value. Applying the logistic sigmoid to the base plus all four contributions reconstructs the score within an absolute tolerance of `0.000002`. The pipeline checks this for every row before writing a run. Explanations describe the model's decision and carry the same limitations as the sampled graph and its partial labels. [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) · [LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)

## Inspect the experiment in MLflow

`elliptic-track --model-run <directory>` imports a completed training comparison. Add `--score-run <directory>` to include the aggregate batch results. The tracker checks that both directories refer to the same training manifest and model. It logs both candidates' validation AP, the selected model's operating point, original test metrics, queue counts and, when available, the batch SHAP importance. The training manifest artifact holds the feature definitions, label policy, hyperparameters and dependency versions.

The default store is `artifacts/tracking/mlflow.db`, with local aggregate artifacts next to it. Reimporting identical reports returns the existing finished run ID. Use `--tracking-dir` to choose another local store. To browse the default store, run `mlflow server --backend-store-uri sqlite:///artifacts/tracking/mlflow.db --host 127.0.0.1 --port 5000` from the project root and open `http://127.0.0.1:5000`. Stop the local server with Ctrl+C. [MLflow backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)

The score table and review queue contain transaction-level records. Keep both files in local, Git-ignored storage. The tracker copies only the training manifest, aggregate training metrics, optional batch summary and aggregate SHAP plot. A replay of the Phase 2 test window uses the same observations and provides no additional independent performance measurement.

## Recorded replay

The saved Phase 2 XGBoost pipeline scored 46,647 transactions from steps 40–49. The validation threshold flagged 19,036. The capacity rule selected 500, with 50 in each step. The largest mean absolute SHAP contribution across the batch came from the number of distinct observed funding transactions (0.491 log-odds), followed by observed ancestor depth (0.134). The maximum absolute difference between the model score and the score reconstructed from SHAP was below 0.00000013. [Original experiment and dataset fingerprint](evaluation.md#recorded-local-result)
