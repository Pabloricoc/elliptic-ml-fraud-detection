import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from elliptic_fraud import train
from elliptic_fraud.api import create_app
from elliptic_fraud.ingestion import EllipticDataset
from elliptic_fraud.schema import DatasetError
from elliptic_fraud.scoring import score_batch


@pytest.fixture
def training(model_input: Path, tmp_path: Path) -> Path:
    return train.run_training(model_input, tmp_path / "training")


@pytest.fixture
def graph_payload(periods: EllipticDataset) -> dict:
    ids = periods.transactions.loc[periods.transactions["time_step"].eq(40), "tx_id"]
    edges = periods.edges.loc[periods.edges["target_id"].isin(ids)]
    return {
        "transaction_id": ids.iloc[2],
        "transactions": ids.tolist(),
        "funding_edges": edges.to_dict("records"),
    }


@pytest.mark.parametrize("winner", ["logistic_regression", "xgboost"])
def test_api_matches_batch_score_and_decision_for_both_models(
    model_input: Path,
    graph_payload: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
) -> None:
    candidates = train._candidates()
    monkeypatch.setattr(train, "_candidates", lambda: {winner: candidates[winner]})
    model_run = train.run_training(model_input, tmp_path / "training")
    manifest = json.loads((model_run / "manifest.json").read_text(encoding="utf-8"))
    scored = score_batch(
        model_run, model_input, tmp_path / "scoring", first_step=40, last_step=40
    )
    batch = pd.read_csv(
        scored / "scores.csv", dtype={"tx_id": str}, float_precision="round_trip"
    ).set_index("tx_id")
    monkeypatch.setenv("ELLIPTIC_MODEL_RUN", str(model_run))
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {
            "status": "ready",
            "model": winner,
            "model_sha256": manifest["model_sha256"],
        }
        for tx_id in graph_payload["transactions"]:
            result = client.post(
                "/v1/score", json={**graph_payload, "transaction_id": tx_id}
            )
            assert result.status_code == 200, result.text
            actual = result.json()
            assert actual["transaction_id"] == tx_id
            assert actual["risk_score"] == pytest.approx(
                batch.loc[tx_id, "risk_score"], abs=1e-7, rel=0
            )
            assert actual["above_threshold"] == batch.loc[tx_id, "above_threshold"]
            assert actual["threshold"] == manifest["threshold"]
            assert actual["model_sha256"] == manifest["model_sha256"]
        assert client.get("/openapi.json").status_code == 200


def test_invalid_graphs_are_rejected_and_outgoing_edges_preserve_the_score(
    training: Path, graph_payload: dict
) -> None:
    a, b, target, _ = graph_payload["transactions"]
    with TestClient(create_app(training)) as client:
        original = client.post("/v1/score", json=graph_payload)
        assert original.status_code == 200
        future = {
            **graph_payload,
            "transactions": [*graph_payload["transactions"], "9999"],
        }
        future["funding_edges"] = [
            *graph_payload["funding_edges"],
            {"source_id": target, "target_id": "9999"},
        ]
        assert client.post("/v1/score", json=future).json() == original.json()
        for bad in [
            {"class": "unknown"},
            {"transaction_id": 4},
            {"transaction_id": "9999"},
            {"transactions": [a, a, target]},
            {"funding_edges": [{"source_id": "9999", "target_id": target}]},
            {
                "funding_edges": [
                    {"source_id": a, "target_id": b},
                    {"source_id": b, "target_id": a},
                ]
            },
        ]:
            assert (
                client.post("/v1/score", json={**graph_payload, **bad}).status_code
                == 422
            )


def test_api_startup_rejects_a_missing_or_corrupted_artifact(
    training: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELLIPTIC_MODEL_RUN", raising=False)
    with (
        pytest.raises(RuntimeError, match="ELLIPTIC_MODEL_RUN"),
        TestClient(create_app()),
    ):
        pytest.fail("The service started without a configured model")
    artifact = training / "model.joblib"
    artifact.unlink()
    with pytest.raises(FileNotFoundError), TestClient(create_app(training)):
        pytest.fail("The service started without a model")
    artifact.write_bytes(b"invalid content")
    with pytest.raises(DatasetError, match="SHA-256"), TestClient(create_app(training)):
        pytest.fail("The service started with a corrupted model")
