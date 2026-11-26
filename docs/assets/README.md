# Portfolio figure provenance

The banner is an original SVG. Its nodes and edges illustrate a funding graph conceptually. The local Python badge records the versions supported by the package and checked in CI.

| Figure | Aggregate source |
| --- | --- |
| `temporal-coverage.png` | `time_steps.csv` from the full-data audit, with the split ranges from the training report. |
| `review-queue.png` | `test_by_step.csv` from the selected model's final-window evaluation. The bars show the known illicit, known licit and unlabelled selections for each step. |
| `shap-importance.png` | Mean absolute feature contributions in `summary.json` from the batch replay of the frozen XGBoost model. |
| `fastapi-openapi.png` | Swagger UI captured from the locally running API. The screenshot shows the public request contract. |

The figures describe the full local Elliptic snapshot identified in the evaluation protocol. The PNGs contain aggregate statistics. The original CSVs, model artifact, score and review rows and MLflow store stay in local ignored directories. [Recorded evaluation](../evaluation.md#recorded-local-result) · [Recorded batch replay](../scoring.md#recorded-replay)

## Rebuild the figures

Create the audit, training and scoring outputs using the project commands. Substitute the three completed run directories below from the same dataset snapshot. Install the pinned development environment before running the script. [Reproduce the pipeline](../../README.md#reproduce-the-pipeline)

```sh
python docs/assets/build_figures.py "artifacts/eda/eda-<run-id>" "artifacts/models/model-<run-id>" "artifacts/scores/score-<run-id>"
```

The script writes the three PNGs next to itself by default. Use `--output-dir` to preview regenerated plots elsewhere.
