"""Start the API image with an invented model and verify one real HTTP prediction."""

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from elliptic_fraud.inference import predict_scores
from elliptic_fraud.model_features import FEATURE_AUDIT, ancestry_features
from elliptic_fraud.runs import file_digest, write_json

TRANSACTIONS = ["0001", "0002", "0003", "0004"]
EDGES = [
    {"source_id": "0001", "target_id": "0003"},
    {"source_id": "0002", "target_id": "0004"},
    {"source_id": "0003", "target_id": "0004"},
]
TARGET = "0004"
THRESHOLD = 0.55


def docker(*args: str, timeout: int = 90) -> str:
    return subprocess.check_output(
        ["docker", *args], text=True, timeout=timeout
    ).strip()


def fetch_json(url: str, body: dict | None = None) -> dict:
    request = (
        Request(url)
        if body is None
        else Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def smoke_test(image: str) -> None:
    """Compare the container prediction to the same synthetic pipeline on the host."""
    features = ancestry_features(pd.Index(TRANSACTIONS), pd.DataFrame(EDGES))
    pipeline = Pipeline(
        [
            (
                "classifier",
                XGBClassifier(
                    n_estimators=8,
                    max_depth=2,
                    min_child_weight=0,
                    n_jobs=1,
                    random_state=42,
                ),
            )
        ]
    ).fit(features, [0, 0, 1, 1])
    expected = float(predict_scores(pipeline, features).loc[TARGET])
    with tempfile.TemporaryDirectory(prefix="elliptic-docker-") as temp:
        run = Path(temp).resolve() / "synthetic-model"
        run.mkdir(mode=0o755)
        artifact = run / "model.joblib"
        joblib.dump(pipeline, artifact, compress=3)
        digest = file_digest(artifact)
        write_json(
            run / "manifest.json",
            {
                "winner": "xgboost",
                "features_in_order": FEATURE_AUDIT,
                "threshold": THRESHOLD,
                "model_sha256": digest,
            },
        )

        container = f"elliptic-smoke-{uuid4().hex}"
        try:
            # Name the container before starting it so that cleanup can also
            # remove a container that never got past the Created state.
            docker(
                "create",
                "--name",
                container,
                "--read-only",
                "--publish",
                "127.0.0.1::8000",
                "--mount",
                f"type=bind,source={run},target=/model,readonly",
                image,
                timeout=180,
            )
            docker("start", container, timeout=180)
            address = docker("port", container, "8000/tcp")
            root = f"http://{address}"
            deadline = time.monotonic() + 150
            while (
                docker("inspect", "--format", "{{.State.Health.Status}}", container)
                != "healthy"
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError("The container did not become healthy.")
                time.sleep(1)
            health = fetch_json(root + "/health")
            if health != {
                "status": "ready",
                "model": "xgboost",
                "model_sha256": digest,
            }:
                raise RuntimeError("The service reported unexpected model metadata.")
            result = fetch_json(
                root + "/v1/score",
                {
                    "transaction_id": TARGET,
                    "transactions": TRANSACTIONS,
                    "funding_edges": EDGES,
                },
            )
            if (
                result["transaction_id"] != TARGET
                or result["model_sha256"] != digest
                or result["threshold"] != THRESHOLD
                or result["above_threshold"] != (expected >= THRESHOLD)
                or not math.isclose(
                    result["risk_score"], expected, abs_tol=1e-7, rel_tol=0
                )
            ):
                raise RuntimeError("The API response differs from the host model.")
            print("Container became healthy and scored the synthetic graph correctly.")
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            RuntimeError,
            URLError,
        ):
            subprocess.run(["docker", "logs", container], check=False, timeout=15)
            raise
        finally:
            subprocess.run(
                ["docker", "rm", "--force", container], check=False, timeout=30
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="tag of the local inference image")
    smoke_test(parser.parse_args().image)


if __name__ == "__main__":
    main()
