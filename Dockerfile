# Use a lightweight Python 3.11 base image
# "slim" reduces image size while keeping compatibility
FROM python:3.11-slim

# -----------------------------
# Python & Poetry runtime flags
# -----------------------------
# PYTHONDONTWRITEBYTECODE: prevent creation of .pyc files
# PYTHONUNBUFFERED: ensure logs are flushed immediately (important for Docker logs)
# POETRY_VIRTUALENVS_CREATE=false: install dependencies into the system environment
# PIP_NO_CACHE_DIR: avoid caching wheels to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

# -----------------------------
# Application working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# System-level dependencies
# -----------------------------
# curl: used by some Python tooling and debugging
# git: required for some Poetry / dependency resolution cases
# Remove apt cache to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Poetry
# -----------------------------
# Poetry is used as the dependency manager
RUN pip install --no-cache-dir poetry

# ---------------------------------------------------
# Copy dependency definition files (layer caching)
# ---------------------------------------------------
# README.md is required because pyproject.toml references it
# Copying these first allows Docker to cache dependency layers
COPY pyproject.toml poetry.lock README.md /app/

# -----------------------------
# Network robustness for pip
# -----------------------------
# Increase timeout and retries to handle slow or unstable networks
ENV PIP_DEFAULT_TIMEOUT=200 \
    PIP_RETRIES=10

# ---------------------------------------------------
# Install runtime dependencies only (no dev, no root)
# ---------------------------------------------------
# installer.max-workers=1 avoids memory spikes during install
# --only main: install only production dependencies
# --no-root: do not install the project package itself
RUN poetry config installer.max-workers 1 \
    && poetry install --no-interaction --no-ansi --only main --no-root


########################################################################
# Hugging Face configuration & model preloading
########################################################################

# Increase Hugging Face download timeout and disable telemetry
ENV HF_HUB_DOWNLOAD_TIMEOUT=400 \
    HF_HUB_DISABLE_TELEMETRY=1

# Explicit cache locations to ensure model persistence inside the image
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# ---------------------------------------------------
# Pre-download Hugging Face model (gpt2)
# ---------------------------------------------------
# This avoids a long cold start on the first container run
# Model weights are downloaded at build time
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('gpt2'); \
AutoModelForCausalLM.from_pretrained('gpt2')"

########################################################################


# -----------------------------
# Copy application source code
# -----------------------------
COPY . /app

# -----------------------------
# Python import configuration
# -----------------------------
# Required because the project uses a src/ layout
ENV PYTHONPATH=/app/src

# -----------------------------
# API exposure
# -----------------------------
EXPOSE 8000

# -----------------------------
# Application entrypoint
# -----------------------------
# Run FastAPI using Uvicorn
# --host 0.0.0.0 allows access from outside the container
CMD ["uvicorn", "llm_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]