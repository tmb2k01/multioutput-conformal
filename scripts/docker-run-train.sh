#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-train}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${PROJECT_ROOT}/.wandb_api_key}"

declare -a WANDB_MOUNT=()
if [[ -f "${WANDB_KEY_FILE}" ]]; then
  WANDB_MOUNT+=("-v" "${WANDB_KEY_FILE}:/app/.wandb_api_key:ro")
else
  echo "No W&B API key found at ${WANDB_KEY_FILE}; continuing without it."
fi

mkdir -p \
  "${PROJECT_ROOT}/data" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/lightning_logs" \
  "${PROJECT_ROOT}/wandb"

WANDB_ARGS=()
if (( ${#WANDB_MOUNT[@]} )); then
  WANDB_ARGS=("${WANDB_MOUNT[@]}")
fi

chmod -R u+rw "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/lightning_logs" "${PROJECT_ROOT}/wandb" || true


USER_FLAGS=()
if [[ -n "${RUN_AS_HOST_UID:-}" ]]; then
  USER_FLAGS=(--user "$(id -u)":"$(id -g)")
fi


echo "Running training from image '${IMAGE_TAG}' (container: ${CONTAINER_NAME})"
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -e WANDB_DIR=/app/wandb \
  -e WANDB_CACHE_DIR=/app/wandb/cache \
  -e WANDB_CONFIG_DIR=/app/wandb/config \
  "${USER_FLAGS[@]}" \
  -v "${PROJECT_ROOT}/data:/app/data" \
  -v "${PROJECT_ROOT}/models:/app/models" \
  -v "${PROJECT_ROOT}/static:/app/static:ro" \
  -v "${PROJECT_ROOT}/lightning_logs:/app/lightning_logs" \
  -v "${PROJECT_ROOT}/wandb:/app/wandb" \
  "${WANDB_ARGS[@]}" \
  ${DOCKER_RUN_ARGS:-} \
  "${IMAGE_TAG}" \
  --train
