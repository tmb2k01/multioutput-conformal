FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG NB_USER=app
ARG NB_UID=1000
ARG TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    WANDB_DIR=/app/wandb WANDB_CACHE_DIR=/app/wandb/cache WANDB_CONFIG_DIR=/app/wandb/config \
    WANDB_API_KEY_FILE=/app/.wandb_api_key \
    PATH="/home/${NB_USER}/.local/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN mkdir -p /app/wandb && chown -R ${NB_UID}:${NB_UID} /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    ca-certificates \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel && \
    groupadd --gid ${NB_UID} ${NB_USER} && \
    useradd --uid ${NB_UID} --gid ${NB_UID} --create-home --shell /bin/bash ${NB_USER} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN if [ -n "${TORCH_EXTRA_INDEX_URL}" ]; then \
    python -m pip install --no-cache-dir --extra-index-url ${TORCH_EXTRA_INDEX_URL} -r requirements.txt; \
    else \
    python -m pip install --no-cache-dir -r requirements.txt; \
    fi


# Pre-download ResNet50 weights
ENV TORCH_HOME=/home/app/.cache/torch
RUN mkdir -p ${TORCH_HOME}/hub/checkpoints \
    && wget -q https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
    -O ${TORCH_HOME}/hub/checkpoints/resnet50-11ad3fa6.pth \
    && chown -R app:app /home/app/.cache

COPY src ./src
COPY static ./static
COPY scripts ./scripts
COPY README.md ./README.md

RUN mkdir -p /app/data /app/models /app/lightning_logs /app/wandb && \
    chown -R ${NB_UID}:${NB_UID} /app

USER ${NB_USER}

ENV PYTHONPATH=/app

EXPOSE 7860

ENTRYPOINT ["python", "-m", "src.main"]
