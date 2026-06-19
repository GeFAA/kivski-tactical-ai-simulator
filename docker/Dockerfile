# Kivski 24/7 trainer image for RunPod (or any CUDA 12.1 host).
#
# Build:
#   docker build -t YOURUSER/kivski:latest \
#       --build-arg REPO_URL=https://github.com/GeFAA/kivski-tactical-ai-simulator.git \
#       --build-arg REPO_REF=main \
#       -f docker/Dockerfile docker/
#
# The image clones the repo at build time so a fresh `docker run` starts
# training immediately. The entrypoint can also fast-forward to a newer
# REPO_REF on container start.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- system deps -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        git \
        ca-certificates \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

# ---- torch (CUDA 12.1) FIRST so the editable install picks it up -----------
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision

# ---- clone repo ------------------------------------------------------------
ARG REPO_URL=https://github.com/GeFAA/kivski-tactical-ai-simulator.git
ARG REPO_REF=main
ENV REPO_URL=${REPO_URL} \
    REPO_REF=${REPO_REF}

WORKDIR /workspace
RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" kivski

WORKDIR /workspace/kivski

# ---- install the monorepo (single root pyproject covers sim+agents+api) ----
# `[cloud]` extra pulls huggingface_hub for HF Hub model/metrics sync.
RUN pip install -e ".[cloud]"

# ---- entrypoints -----------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY 24-7-supervisor.sh /usr/local/bin/24-7-supervisor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/24-7-supervisor.sh

# Persistent volume for checkpoints / restart history
RUN mkdir -p /workspace/persistent /workspace/restart-history

# Default: supervisor wraps entrypoint with restart-storm protection.
ENTRYPOINT ["/usr/local/bin/24-7-supervisor.sh"]
