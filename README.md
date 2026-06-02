# Master's Thesis – Multi-Output Classification with Clustered Conformal Prediction

## Project Overview

This repository contains the code and experimental framework for my master's thesis on **Implementation of Multi-Output Classification with Clustered Conformal Prediction**, aiming to enhance the reliability of structured prediction tasks. The focus is on quantifying uncertainty in predictions using conformal methods, particularly in the presence of output dependencies, and evaluating their effectiveness through reproducible experiments.

The project addresses the following key areas:

- **Multi-output classification**: Tackling problems where each input is associated with multiple interdependent labels.
- **Clustered conformal prediction**: Improving prediction sets by accounting for structure within output spaces.

## Documentation

For detailed information about specific components, refer to the following documentation:

- [Data Acquisition](doc/data-acquisition.md): Downloading, organizing, and splitting the dataset for experiments.
- [Dockerization Overview](doc/docker-overview.md): Container layout, volume mounts, and configurable environment variables.
- [Docker Workflows](doc/docker-workflows.md): Step-by-step instructions for building the image and running tasks in Docker.
- [Model Definition](doc/model-definition.md): Architecture and implementation details of the models.
- [Conformal Prediction](doc/conformal-prediction.md): Overview of the conformal prediction methodology.
- [Metrics](doc/metrics.md): Evaluation metrics and performance analysis.
- [Web Interface](doc/web-interface.md): Guide to using the web-based prediction interface.

## How to Start

### Without Docker

- **Prerequisites:** Python 3.12+, `pip`, and Git.
- **Clone the repository:**

  ```bash
  git clone https://github.com/tmb2k01/masters-thesis.git
  cd masters-thesis
  ```

- **(Recommended) Create a virtual environment:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

- **Install dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

- **Prepare datasets locally:**

  ```bash
  bash scripts/prepare-data.sh
  ```

- **Run a workflow** (the CLI exposes one subcommand per task; run from `src/`,
  i.e. with `PYTHONPATH=src`):

  ```bash
  python3 -m main experiment experiments/utkface/hinge/ll_ll_cal.yaml  # full run + metrics
  python3 -m main train experiments/utkface/hinge/ll_ll_cal.yaml       # train model only
  python3 -m main calibrate experiments/utkface/hinge/ll_ll_cal.yaml   # calibrate only
  ```

  Store your Weights & Biases API key in `.wandb_api_key` (or point the
  `WANDB_API_KEY_FILE` environment variable elsewhere) to enable experiment tracking.

- **Serve the web interface:**

  ```bash
  python3 -m main web_service              # uses static/ws-config.json by default
  python3 -m main web_service --config path/to/ws-config.json
  ```

  The Gradio UI listens on `http://localhost:7860` by default. Set `ARTIFACTS_ROOT`
  to point at the directory holding the trained artifacts.

### With Docker

- **Prerequisites:** Docker (24+) and, optionally, the NVIDIA Container Toolkit for GPU support.
- **Build the image:**

  ```bash
  scripts/docker-build.sh
  ```

- **Prepare data inside the container (optional):**

  ```bash
  scripts/docker-prepare-data.sh
  ```

- **Launch training / calibration:**

  ```bash
  scripts/docker-run-train.sh [HOST_DATA_DIR] [HOST_ARTIFACTS_DIR] [CONFIG]
  scripts/docker-run-calibrate.sh [HOST_DATA_DIR] [HOST_ARTIFACTS_DIR] [CONFIG]
  ```

  `CONFIG` defaults to `experiments/utkface/hinge/ll_ll_cal.yaml`. Override host
  directories or enable GPU support by exporting `DOCKER_RUN_ARGS`,
  `RUN_AS_HOST_UID`, and the other environment variables documented in
  `doc/docker-workflows.md`.

- **Start the web interface:**

  ```bash
  scripts/docker-run-web.sh [HOST_ARTIFACTS_DIR]
  ```

  Set `PORT=7860` (or any free port) to control the host binding. Artifacts are
  read from the mounted `artifacts` directory (`ARTIFACTS_ROOT=/app/artifacts`).

For detailed explanations of the container layout, volume mounts, and environment
variables, consult the Docker documentation linked above.

## Project Structure

- `src/`: Source code for the implementation
  - `main.py`: CLI entrypoint (`experiment`, `train`, `calibrate`, `web_service`)
  - `experiment.py`: Experiment, training, and calibration orchestration
  - `metrics.py`: Conformal-prediction evaluation metrics
  - `web_service.py`: Gradio web interface
  - `core/`: Models, calibrators, the `ConformalPredictor`, and supporting types
  - `calibration/`: Nonconformity scores, conformal prediction, and calibration utilities
  - `data/`: Dataset handling and preprocessing
- `experiments/`: Experiment/train/calibrate YAML configs
- `scripts/`: Data preparation and Docker helper scripts
- `notebooks/`: Jupyter notebooks for analysis and experiments
- `doc/`: Detailed documentation
- `static/`: Configuration files and web assets (`ws-config.json`, `styles.css`)
