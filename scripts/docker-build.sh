#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-masters-thesis:latest}"
BUILD_ARG="${TORCH_EXTRA_INDEX_URL-https://download.pytorch.org/whl/cu124}"

echo "Building Docker image '${IMAGE_TAG}' from ${PROJECT_ROOT}"
docker build \
  --file "${PROJECT_ROOT}/Dockerfile" \
  --build-arg "TORCH_EXTRA_INDEX_URL=${BUILD_ARG}" \
  --tag "${IMAGE_TAG}" \
  "${PROJECT_ROOT}"
