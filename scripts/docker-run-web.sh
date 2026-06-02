#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-web}"
PORT="${PORT:-80}"
DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS:---shm-size=2g --ipc=host}"

HOST_ARTIFACTS_DIR="${1:-${PROJECT_ROOT}/artifacts}"
WS_CONFIG="${WS_CONFIG:-./static/ws-config.json}"
mkdir -p "${HOST_ARTIFACTS_DIR}"

# Detect docker GPU runtime
GPU_ARGS=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi '"nvidia"'; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_ARGS=(--gpus all)
  fi
fi

APP_ENVS=(
  -e ARTIFACTS_ROOT=/app/artifacts
)

echo "Starting web service from image '${IMAGE_TAG}' (container: ${CONTAINER_NAME})"
echo "Host artifacts dir: ${HOST_ARTIFACTS_DIR}"
echo "GPU args: ${GPU_ARGS[*]:-(none)}"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  ${DOCKER_RUN_ARGS} \
  -p "${PORT}:7860" \
  "${APP_ENVS[@]}" \
  -v "${HOST_ARTIFACTS_DIR}:/app/artifacts:ro" \
  -v "${PROJECT_ROOT}/static:/app/static:ro" \
  "${IMAGE_TAG}" \
  web_service --config "${WS_CONFIG}"
