#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-train}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${PROJECT_ROOT}/.wandb_api_key}"
DOCKER_RUN_ARGS="--shm-size=8g --ipc=host"

# Ensure host paths exist
mkdir -p \
  "${PROJECT_ROOT}/data" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/lightning_logs" \
  "${PROJECT_ROOT}/wandb"

# Make host dirs writable by current user
chown -R "$(id -u)":"$(id -g)" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/lightning_logs" \
  "${PROJECT_ROOT}/wandb" 2>/dev/null || true
chmod -R u+rwX "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/lightning_logs" "${PROJECT_ROOT}/wandb" || true

# Optional: run container with host UID/GID to avoid permission issues on mounted dirs
USER_FLAGS=()
if [[ "${RUN_AS_HOST_UID:-1}" != "0" ]]; then
  USER_FLAGS=(--user "$(id -u)":"$(id -g)")
fi

# W&B configuration
WANDB_ARGS=(
  -e WANDB_MODE=online
  -e WANDB_DIR=/app/wandb
  -e WANDB_CACHE_DIR=/app/wandb/cache
  -e WANDB_CONFIG_DIR=/app/wandb/config
)
if [[ -f "${WANDB_KEY_FILE}" ]]; then
  # Provide key via env and also mount the file read-only (optional)
  WANDB_ARGS+=(-e "WANDB_API_KEY=$(<"${WANDB_KEY_FILE}")")
  WANDB_ARGS+=(-v "${WANDB_KEY_FILE}:/app/.wandb_api_key:ro")
else
  echo "No W&B API key found at ${WANDB_KEY_FILE}; W&B will prompt or fail to authenticate."
fi

echo "Running training from image '${IMAGE_TAG}' (container: ${CONTAINER_NAME})"
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  "${USER_FLAGS[@]}" \
  "${WANDB_ARGS[@]}" \
  -v "${PROJECT_ROOT}/data:/app/data:ro" \
  -v "${PROJECT_ROOT}/models:/app/models:rw" \
  -v "${PROJECT_ROOT}/static:/app/static:ro" \
  -v "${PROJECT_ROOT}/lightning_logs:/app/lightning_logs:rw" \
  -v "${PROJECT_ROOT}/wandb:/app/wandb:rw" \
  ${DOCKER_RUN_ARGS} \
  "${IMAGE_TAG}" \
  --train
