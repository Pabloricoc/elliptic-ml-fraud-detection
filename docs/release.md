# First GitHub release

`v1.0.0` is the proposed first GitHub release of the completed portfolio project. Tag the reviewed documentation and code together on `main` after GitHub Actions succeeds on that commit. The Python distribution is currently version `0.1.0` in [`pyproject.toml`](../pyproject.toml). A package release or version bump is a separate change.

## Release candidate checklist

- [ ] Merge the portfolio documentation PR after review and confirm the workflow passes on the merge commit.
- [ ] Confirm the repository's code licensing terms before describing the source as open source. Dataset access and reuse terms are recorded separately. [Dataset licence and provenance](../data/README.md#licence-and-provenance)
- [ ] Create the `v1.0.0` tag at the verified commit and publish a GitHub release with the notes below. GitHub supplies the tagged source archives. Model and data artifacts stay local. [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

## Draft release notes

**Elliptic ML Fraud Detection v1.0.0**

The initial project release covers schema validation and EDA for the original Elliptic graph, chronological model selection, batch scoring with SHAP and MLflow, and a FastAPI demonstration packaged for Docker. It includes cross-platform tests and a synthetic API smoke test in GitHub Actions.

In the recorded test window, the selected XGBoost model achieved **0.0882 average precision** on labelled transactions. The fixed-budget queue captured **5 of 636 known illicit cases**. These figures describe the documented historical split. The released graph has partial label coverage, and the service targets local demonstration. [Evaluation protocol and results](evaluation.md#recorded-local-result)

Start with the [README](../README.md#reproduce-the-pipeline) to reproduce the experiment and the [serving guide](serving.md) to run the local API.
