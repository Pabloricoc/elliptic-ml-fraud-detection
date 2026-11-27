# Holdout access ledger

The holdout is steps 40–49 of the Elliptic release. Record the date, exact code revision when available, frozen candidates, purpose and decision scope for each access, including failed attempts. Phase 1 authorises one replay of the legacy run. Phase 3 authorises one filter-only read to create the development export. The subsequent final comparison requires an approved pre-registration. The period has already been observed, including the known change in outcomes from step 43.

| Date | Code reference | Candidate | Purpose and decision scope |
| --- | --- | --- | --- |
| 2026-11-25 | `edbdc40` records the original implementation and results | Frozen legacy T0 XGBoost | Initial evaluation of the model and threshold chosen on steps 30–39. The exact checkout at the time of the run was not recorded as a Git revision. |
| 2026-11-26 | `f5fc191` and `346e25c` record the batch and API verification | Same frozen T0 XGBoost | Replay and serving parity for the deployed model. This used existing holdout scores and inference inputs. |
| 2026-11-27 | `9248c37` | Same frozen T0 XGBoost | Independent read-only audit computed aggregate diagnostics from the existing score output. |
| 2026-11-27 | `3b6769e93b7fff5892ca89feca92e736d7e3fedd` | Same frozen T0 XGBoost (`model-4e919f27…`) | Re-measurement of legacy run, no selection. Single authorised access completed with exact legacy parity and an exact match to the batch replay. Local aggregate artifact: `artifacts/evaluations/evaluation-c682e47d1ed946f389f5a3129dcbf9f7/evaluation.json`. |
| 2026-11-27 | `a56279370c2032aa69c6c565e38199f023a7162f` | — | filter-only read of the full release to produce the development export; no holdout features, labels, scores or metrics computed |

Phases 3 and 4 use only the verified export at `data/processed/development/`. Keep transaction-level records and scores in ignored local directories.
