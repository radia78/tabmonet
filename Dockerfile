# Use a lightweight Python base image
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --no-install-project --no-dev

COPY . .

RUN uv sync --frozen --no-dev
