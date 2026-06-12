# Lean inference / daily-scheduler image.
# Installs ONLY the CPU torch flavor + the upscale extra — no UI, no train, no
# dev deps, and crucially no multi-GB CUDA stack. For a GPU host, build with
#   --build-arg TORCH_GROUP=cuda
# and a CUDA base image.
FROM python:3.11-slim AS base

ARG TORCH_GROUP=cpu

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    UV_CACHE_DIR=/tmp/uv-cache \
    HF_HOME=/models

# uv for reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY wallgen ./wallgen
COPY config.yaml ./
COPY prompts ./prompts

# Resolve from the lockfile; only the requested torch group (lean inference image).
RUN if [ "$TORCH_GROUP" = "cuda" ]; then \
        uv sync --frozen --no-group cpu --group cuda --no-dev ; \
    else \
        uv sync --frozen --no-dev ; \
    fi

# Persist model downloads across runs by mounting a volume at /models.
VOLUME ["/models", "/app/output"]

ENTRYPOINT ["uv", "run", "python", "-m", "wallgen"]
CMD ["daily"]
