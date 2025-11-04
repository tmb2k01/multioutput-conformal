# Dockerization Overview

This project provides a Docker-based workflow so you can reproduce experiments and
serve the web interface without managing a local Python environment. The container
image bundles the training code, the web UI, and all dependencies required by the
thesis.

## Base Image and Runtime

- The Dockerfile extends `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, matching
  the CUDA runtime expected by the PyTorch wheels listed in `requirements.txt`.
- Python 3.10, PyTorch, and the project dependencies are installed inside the
  image. The ResNet-50 weights used by the models are pre-fetched to speed up
  startup on first run.
- The default entrypoint launches `python -m src.main`, which auto-detects whether
  it should run the training CLI or start the Gradio-powered web UI, depending on
  the passed command-line arguments.

## Data and Model Persistence

The container keeps its working tree in `/app`. The runtime scripts mount the host
directories below to make experiment artefacts persistent between runs:

- `/app/data`: raw datasets and calibration splits (mounted read-only during
  training).
- `/app/models`: trained checkpoints and exported models (mounted read-only for
  the web UI).
- `/app/lightning_logs`: PyTorch Lightning logs from training and validation.
- `/app/wandb`: Weights & Biases cache and run metadata (mounted when available).
- `/app/static`: configuration files and web assets shipped with the repository.

By default, the scripts map these paths to the matching folders in the project
root, but you can override them by passing explicit paths as positional arguments.

## GPU Detection

All Docker helper scripts automatically check whether the Docker daemon exposes an
`nvidia` runtime and whether `nvidia-smi` is available on the host. When both
conditions are met, the container starts with `--gpus all`; otherwise, it falls
back to CPU execution. You can force GPU access by setting
`DOCKER_RUN_ARGS="--gpus all"` or disable it by setting `GPU_ARGS=()` before
invoking the script.

## Configurable Environment Variables

Each script supports a small set of environment variables so you can tailor the
workflow to your infrastructure:

- `IMAGE_TAG`: Docker image tag to run/build (defaults to `masters-thesis:latest`).
- `DOCKER_RUN_ARGS`: Extra flags passed to `docker run` (defaults allocate extra
  shared memory and `--ipc=host`).
- `CONTAINER_NAME`: Friendly container name for each workflow.
- `RUN_AS_HOST_UID`: When set to `0`, the training container keeps the default
  user inside the image; otherwise it matches the host UID/GID to avoid file
  permission issues on mounted volumes.
- `WANDB_KEY_FILE`: Path to a text file containing your Weights & Biases API token
  (used by the training workflow).

Refer to `doc/docker-workflows.md` for task-specific instructions using these
settings.
