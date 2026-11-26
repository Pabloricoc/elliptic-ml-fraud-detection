# Serve the frozen model

The API loads one trusted training directory at startup. It uses the same confirmation-time ancestry features, positive class and validation threshold as batch scoring. The directory must contain the matching `model.joblib` and `manifest.json`. The loader verifies the model checksum, predictor order and class order. Use an artifact you trained or whose source you trust, with the pinned dependency versions.

## Run in Docker Desktop

Use Linux containers and run these commands from the repository root in PowerShell. Set `$modelRun` to the directory produced by `elliptic-train`.

```powershell
$modelRun = (Resolve-Path 'artifacts/models/model-<run-id>').Path
docker build --tag elliptic-fraud:local .
docker run --rm --read-only --publish 127.0.0.1:8000:8000 --mount "type=bind,source=$modelRun,target=/model,readonly" elliptic-fraud:local
```

Open `http://127.0.0.1:8000/docs` to explore the API, or send the request below from a second terminal. Ctrl+C stops the container. The image build context is limited to packaging metadata and Python source. The model is read from the runtime mount. GitHub Actions builds the image and checks HTTP inference using only a synthetic model and graph. [Docker build context](https://docs.docker.com/build/concepts/context/#dockerignore-files) · [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)

If your network uses a private certificate authority for package downloads, provide its trusted PEM bundle at build time with `docker build --secret "id=pip_cert,src=C:\path\to\roots.pem" --tag elliptic-fraud:local .`. BuildKit makes the bundle available to pip during installation. [Docker build secrets](https://docs.docker.com/build/building/secrets/)

For a native run, set `$env:ELLIPTIC_MODEL_RUN = $modelRun` and run `python -m uvicorn elliptic_fraud.api:create_app --factory --host 127.0.0.1 --port 8000` in the installed development environment.

## Score a transaction

| Endpoint | Result |
| --- | --- |
| `GET /health` | Readiness and the loaded model's name and SHA-256. |
| `POST /v1/score` | Illicit-class score, frozen threshold, alert flag and model SHA-256. |
| `GET /docs` | Interactive OpenAPI documentation. |

Use decimal strings for transaction IDs so that leading zeros are retained. Supply unique nodes from the same Elliptic time-step graph, including the transaction being scored and all its observed incoming ancestors. Each directed funding edge connects two supplied nodes. The graph can include unrelated nodes or later outgoing edges, which do not affect the target's ancestry. The service validates structure and cycles. The caller supplies the observed graph context at confirmation time and is responsible for its completeness.

```powershell
$body = @{
    transaction_id = '0004'
    transactions = @('0001', '0002', '0003', '0004')
    funding_edges = @(
        @{ source_id = '0001'; target_id = '0003' }
        @{ source_id = '0002'; target_id = '0004' }
        @{ source_id = '0003'; target_id = '0004' }
    )
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/score -ContentType 'application/json' -Body $body
```

A transaction with no observed parents uses its own ID and an empty edge list. The request accepts up to 10,000 unique nodes and 25,000 edges. Invalid requests return HTTP 422. The score is uncalibrated. The API applies the frozen threshold immediately. Review ranking and the end-of-step capacity remain part of the batch scoring workflow. [Batch scoring guide](scoring.md)

## Verify the container

With the project installed locally, run `python scripts/smoke_container.py elliptic-fraud:local` after building the image. The script creates a temporary synthetic model, mounts it read-only, waits for Docker to report the service healthy and compares an HTTP score to the same pipeline on the host. It removes the test container after a successful check or a failed startup.
