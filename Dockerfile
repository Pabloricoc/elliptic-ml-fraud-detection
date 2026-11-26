FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ELLIPTIC_MODEL_RUN=/model

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md constraints-dev.txt ./
COPY src/elliptic_fraud ./src/elliptic_fraud
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=secret,id=pip_cert \
    if [ -f /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi \
    && python -m pip install --timeout 120 --progress-bar off -c constraints-dev.txt .

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "elliptic_fraud.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
