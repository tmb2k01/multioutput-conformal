#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-web}"
PORT="${PORT:-8080}"

mkdir -p \
  "${PROJECT_ROOT}/data" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/lightning_logs" \
  "${PROJECT_ROOT}/wandb"

echo "Starting web service from image '${IMAGE_TAG}' (container: ${CONTAINER_NAME})"
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${PORT}:8080" \
  -v "${PROJECT_ROOT}/data:/app/data" \
  -v "${PROJECT_ROOT}/models:/app/models" \
  -v "${PROJECT_ROOT}/static:/app/static:ro" \
  -v "${PROJECT_ROOT}/lightning_logs:/app/lightning_logs" \
  -v "${PROJECT_ROOT}/wandb:/app/wandb" \
  ${DOCKER_RUN_ARGS:-} \
  "${IMAGE_TAG}"
