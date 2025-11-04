#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-masters-thesis-prepare-data}"
DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS:---shm-size=8g --ipc=host}"

BASE_DIR="${1:-$PROJECT_ROOT}"

mkdir -p "${BASE_DIR}/data"

echo "Running data preparation inside Docker container '${CONTAINER_NAME}' from image '${IMAGE_TAG}'"
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --entrypoint /bin/bash \
  ${DOCKER_RUN_ARGS} \
  -v "${BASE_DIR}/data:/app/data" \
  -v "${PROJECT_ROOT}/scripts:/app/scripts:ro" \
  -w /app \
  --user root \
  --privileged \
  "${IMAGE_TAG}" \
  -c "touch /app/data/.write_test && rm /app/data/.write_test && ./scripts/prepare-data.sh"
