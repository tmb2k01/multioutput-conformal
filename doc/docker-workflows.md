# Docker Workflows

This guide explains how to build the container image and run the main project
tasks inside Docker. All commands are meant to be executed from the repository
root unless stated otherwise.

## Prerequisites

- Docker Engine 24 or newer
- (Optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  if you want GPU acceleration
- An internet connection the first time you build the image (to download base
  layers, Python wheels, and pretrained weights)

## 1. Build the Image

```bash
scripts/docker-build.sh
```

The script wraps `docker build` and accepts:

- `IMAGE_TAG` to change the target tag (defaults to `masters-thesis:latest`).
- `TORCH_EXTRA_INDEX_URL` to pick a different PyTorch wheel index (set it to an
  empty string if you only need CPU builds).

After the build succeeds you can reuse the same image for data preparation,
training, and the web frontend.

## 2. Prepare Data (Optional)

If you prefer to run data preparation inside Docker, use:

```bash
scripts/docker-prepare-data.sh [HOST_PROJECT_DIR]
```

- When omitted, `HOST_PROJECT_DIR` defaults to the repository root.
- The script mounts `<HOST_PROJECT_DIR>/data` as `/app/data` inside the
  container and executes `scripts/prepare-data.sh`.
- Root privileges are required inside the container so the script runs with a
  temporary `root` user to create and clean up files.

You can still run `scripts/prepare-data.sh` directly on the host if you already
have Python dependencies installed locally.

## 3. Run Training Jobs

Launch training with:

```bash
scripts/docker-run-train.sh [HOST_DATA_DIR] [HOST_MODELS_DIR]
```

- `HOST_DATA_DIR` and `HOST_MODELS_DIR` default to `./data` and `./models`.
- Additional directories (`./lightning_logs`, `./wandb`) are created as needed
  so checkpoints and logging artefacts stay on the host.
- Set `RUN_AS_HOST_UID=0` if you prefer to leave files owned by the container
  user; otherwise the script aligns the UID/GID with the host to avoid permission
  issues.
- Store your Weights & Biases API key in `.wandb_api_key` or point
  `WANDB_KEY_FILE` at a custom path to automatically enable W&B tracking.
- Append extra CLI options after `--` to forward arguments to `python -m src.main`.

Example with GPU access and a custom data directory:

```bash
RUN_AS_HOST_UID=1 DOCKER_RUN_ARGS="--shm-size=16g --ipc=host --gpus all" \
  scripts/docker-run-train.sh /mnt/datasets/masters-thesis ./models
```

## 4. Serve the Web Interface

Start the Gradio-based web UI on `http://localhost:7860`:

```bash
PORT=7860 scripts/docker-run-web.sh [HOST_MODELS_DIR]
```

- `HOST_MODELS_DIR` defaults to `./models` and is mounted read-only so the web
  service can load trained checkpoints.
- The script exposes port `7860` from the container but allows you to remap it by
  setting `PORT` (the value before the colon in `-p "${PORT}:7860"`).
- GPU support is detected automatically; override by adding `--gpus all` to
  `DOCKER_RUN_ARGS` when desired.

To stop the service, interrupt the script (Ctrl+C) or run `docker stop` on the
container name shown in the terminal.

## Customising Runs

- Add global Docker flags by extending `DOCKER_RUN_ARGS` (for example,
  `DOCKER_RUN_ARGS="--network host"`).
- Change `IMAGE_TAG` if you maintain multiple variants (e.g., CPU-only and
  GPU-enabled builds).
- Combine scripts with `docker compose` or CI pipelines by reusing the same
  environment variables documented here.

Refer to `doc/docker-overview.md` for architectural details about the image
layout and runtime assumptions.
