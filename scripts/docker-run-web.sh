#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-web}"
PORT="${PORT:-80}"
DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS:---shm-size=2g --ipc=host}"

HOST_MODELS_DIR="${1:-${PROJECT_ROOT}/models}"
mkdir -p "${HOST_MODELS_DIR}"

# Detect docker GPU runtime
GPU_ARGS=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi '"nvidia"'; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_ARGS=(--gpus all)
  fi
fi

APP_ENVS=(
  -e DATA_DIR=/app/data
  -e MODELS_DIR=/app/models
)

echo "Starting web service from image '${IMAGE_TAG}' (container: ${CONTAINER_NAME})"
echo "Host models dir: ${HOST_MODELS_DIR}"
echo "GPU args: ${GPU_ARGS[*]:-(none)}"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  ${DOCKER_RUN_ARGS} \
  -p "${PORT}:7860" \
  "${APP_ENVS[@]}" \
  -v "${HOST_MODELS_DIR}:/app/models:ro" \
  -v "${PROJECT_ROOT}/static:/app/static:ro" \
  "${IMAGE_TAG}"
