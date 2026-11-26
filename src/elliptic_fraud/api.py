"""Serve confirmation-time transaction scores from a frozen local model."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .inference import FrozenModel, load_frozen_model, predict_scores
from .model_features import ancestry_features
from .runs import read_json
from .schema import DatasetError

TransactionId = Annotated[
    str, StringConstraints(strict=True, max_length=128, pattern=r"^[0-9]+$")
]


class FundingEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: TransactionId
    target_id: TransactionId


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: TransactionId
    transactions: list[TransactionId] = Field(min_length=1, max_length=10_000)
    funding_edges: list[FundingEdge] = Field(max_length=25_000)

    @model_validator(mode="after")
    def validate_transactions(self) -> "ScoreRequest":
        ids = set(self.transactions)
        if len(ids) != len(self.transactions):
            raise ValueError("Transaction IDs must be unique.")
        if self.transaction_id not in ids:
            raise ValueError("The scored transaction must appear in transactions.")
        return self


class HealthResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model: str
    model_sha256: str


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    threshold: float
    above_threshold: bool
    model_sha256: str


def create_app(model_run: Path | None = None) -> FastAPI:
    """Create an API that loads one training run at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        directory = model_run
        if directory is None:
            configured = os.environ.get("ELLIPTIC_MODEL_RUN")
            if not configured:
                raise RuntimeError(
                    "Set ELLIPTIC_MODEL_RUN to a trusted training directory."
                )
            directory = Path(configured)
        manifest = read_json(directory / "manifest.json")
        app.state.model = load_frozen_model(directory, manifest)
        yield

    app = FastAPI(title="Elliptic Fraud Scoring", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> HealthResponse:
        model: FrozenModel = request.app.state.model
        return HealthResponse(model=model.name, model_sha256=model.sha256)

    @app.post(
        "/v1/score",
        responses={422: {"description": "Invalid transaction or funding graph"}},
    )
    def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
        model: FrozenModel = request.app.state.model
        edges = pd.DataFrame(
            [(edge.source_id, edge.target_id) for edge in payload.funding_edges],
            columns=["source_id", "target_id"],
        )
        try:
            features = ancestry_features(
                pd.Index(payload.transactions, name="tx_id"), edges
            )
        except DatasetError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        risk = float(
            predict_scores(model.pipeline, features.loc[[payload.transaction_id]]).iloc[
                0
            ]
        )
        return ScoreResponse(
            transaction_id=payload.transaction_id,
            risk_score=risk,
            threshold=model.threshold,
            above_threshold=risk >= model.threshold,
            model_sha256=model.sha256,
        )

    return app
