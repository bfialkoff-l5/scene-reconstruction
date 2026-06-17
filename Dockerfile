FROM python:3.12-slim-bookworm

# OpenCV headless runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
COPY src ./src

RUN uv sync --no-dev

ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "scene_recon.cli"]
