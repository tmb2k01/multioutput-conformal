#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-train}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${PROJECT_ROOT}/.wandb_api_key}"
DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS:---shm-size=8g --ipc=host}"

HOST_DATA_DIR="${1:-${PROJECT_ROOT}/data}"
HOST_MODELS_DIR="${2:-${PROJECT_ROOT}/models}"
HOST_LOGS_DIR="${PROJECT_ROOT}/lightning_logs"
HOST_WANDB_DIR="${PROJECT_ROOT}/wandb"

mkdir -p "${HOST_DATA_DIR}" "${HOST_MODELS_DIR}" "${HOST_LOGS_DIR}" "${HOST_WANDB_DIR}"
chown -R "$(id -u)":"$(id -g)" "${HOST_MODELS_DIR}" "${HOST_LOGS_DIR}" "${HOST_WANDB_DIR}" 2>/dev/null || true
chmod -R u+rwX "${HOST_MODELS_DIR}" "${HOST_LOGS_DIR}" "${HOST_WANDB_DIR}" || true

USER_FLAGS=()
if [[ "${RUN_AS_HOST_UID:-1}" != "0" ]]; then
  USER_FLAGS=(--user "$(id -u)":"$(id -g)")
fi

WANDB_ARGS=(
  -e WANDB_MODE=online
  -e WANDB_DIR=/app/wandb
  -e WANDB_CACHE_DIR=/app/wandb/cache
  -e WANDB_CONFIG_DIR=/app/wandb/config
)
if [[ -f "${WANDB_KEY_FILE}" ]]; then
  WANDB_ARGS+=(-e "WANDB_API_KEY=$(<"${WANDB_KEY_FILE}")")
  WANDB_ARGS+=(-v "${WANDB_KEY_FILE}:/app/.wandb_api_key:ro")
else
  echo "No W&B API key at ${WANDB_KEY_FILE}; W&B may prompt or fail."
fi

APP_ENVS=(
  -e DATA_DIR=/app/data
  -e MODELS_DIR=/app/models
)

echo "Running training from image '${IMAGE_TAG}' (container: ${CONTAINER_NAME})"
echo "Host data dir  : ${HOST_DATA_DIR}"
echo "Host models dir: ${HOST_MODELS_DIR}"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  ${DOCKER_RUN_ARGS} \
  "${USER_FLAGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${APP_ENVS[@]}" \
  -v "${HOST_DATA_DIR}:/app/data:ro" \
  -v "${HOST_MODELS_DIR}:/app/models:rw" \
  -v "${PROJECT_ROOT}/static:/app/static:ro" \
  -v "${HOST_LOGS_DIR}:/app/lightning_logs:rw" \
  -v "${HOST_WANDB_DIR}:/app/wandb:rw" \
  "${IMAGE_TAG}" \
  --train
