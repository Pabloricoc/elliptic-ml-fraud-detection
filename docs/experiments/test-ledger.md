# Holdout access ledger

The holdout is steps 40–49 of the Elliptic release. Record the date, exact code revision when available, frozen candidates, purpose and decision scope for each access, including failed attempts. Phase 1 authorises one replay of the legacy run. The subsequent final comparison requires an approved pre-registration. The period has already been observed, including the known change in outcomes from step 43.

| Date | Code reference | Candidate | Purpose and decision scope |
| --- | --- | --- | --- |
| 2026-11-25 | `edbdc40` records the original implementation and results | Frozen legacy T0 XGBoost | Initial evaluation of the model and threshold chosen on steps 30–39. The exact checkout at the time of the run was not recorded as a Git revision. |
| 2026-11-26 | `f5fc191` and `346e25c` record the batch and API verification | Same frozen T0 XGBoost | Replay and serving parity for the deployed model. This used existing holdout scores and inference inputs. |

The technical audit inspected previously produced holdout outputs. Future development decisions use only steps 1–39. Keep transaction-level records and scores in ignored local directories.
