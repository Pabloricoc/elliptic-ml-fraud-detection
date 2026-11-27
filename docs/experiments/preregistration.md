# Final evaluation pre-registration template

Status: template. Fill in and commit an owner-approved record before the one final evaluation on steps 40–49.

## Shortlist and provenance

- Record the approval date, Git commit, dataset snapshot SHA-256, feature-code hashes, dependency versions and run seeds.
- Register the legacy T0 XGBoost reference and up to four candidates. Include the best eligible S1 and S2 candidates. Record each scenario, feature tier and version, family, exact parameters and development report hashes.
- Record the feature availability evidence. S1 uses confirmation-time ancestry tiers T0/T1. Report S2 as end-of-step triage conditional on A1. Any expansion of the S1 contract requires Phase 3 evidence and separate owner approval.

## Decisions fixed on development

- Rank candidates by mean step AP across F1–F4. Report paired fold differences against the current baseline. A candidate improves if the difference is positive in at least three of four folds and its mean exceeds one paired standard error. Within one paired standard error, prefer the simpler model.
- Require a cutoff tie group of at most five transactions in at least 90% of validation steps at operational capacity 50. Require mean step P@50 on known labels to meet or exceed the current baseline. Record the optional two-step embargo sensitivity if run.
- Specify the final fit on steps 1–39, including the median of fold best iterations for early-stopped models and the secondary threshold determined on F4 out-of-fold scores. Record the frozen model and manifest digests before evaluating. Keep top-50 per step as the operating policy.

## Final report and deployment decision

- Record the approved evaluation command and one test access in the [ledger](test-ledger.md). Report every registered candidate, including regressions.
- Use the [evaluation protocol](../evaluation.md) for support, AP, ROC-AUC, Brier score, operating point, Q-lab, Q-all, random expectation, ties, whole-step bootstrap intervals and the descriptive step-43 split. Report paired comparisons against the frozen reference.
- Record the owner-approved S1 deployment criteria and verification. The legacy model remains the deployed reference pending the final evaluation and an explicit owner decision. S2 remains batch-only.
