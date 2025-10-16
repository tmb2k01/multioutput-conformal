# Master's Thesis – Multi-Output Classification with Clustered Conformal Prediction

## Project Overview

This repository contains the code and experimental framework for my master's thesis on **Implementation of Multi-Output Classification with Clustered Conformal Prediction**, aiming to enhance the reliability of structured prediction tasks. The focus is on quantifying uncertainty in predictions using conformal methods, particularly in the presence of output dependencies, and evaluating their effectiveness through reproducible experiments.

The project addresses the following key areas:

- **Multi-output classification**: Tackling problems where each input is associated with multiple interdependent labels.
- **Clustered conformal prediction**: Improving prediction sets by accounting for structure within output spaces.

## Documentation

For detailed information about specific components, refer to the following documentation:

- [Data Acquisition](doc/data-acquisition.md): Downloading, organizing, and splitting the dataset for experiments.
- [Model Definition](doc/model-definition.md): Architecture and implementation details of the models.
- [Conformal Prediction](doc/conformal-prediction.md): Overview of the conformal prediction methodology.
- [Metrics](doc/metrics.md): Evaluation metrics and performance analysis.
- [Web Interface](doc/web-interface.md): Guide to using the web-based prediction interface.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/tmb2k01/masters-thesis.git
cd masters-thesis
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare the datasets:

```bash
bash scripts/prepare-data.sh
```

### Docker Workflow

The repository includes a GPU-ready Docker image so you can run training or the web UI without managing local Python environments.

1. Build the image (override `TORCH_EXTRA_INDEX_URL` with an empty string if you only need CPU wheels):

   ```bash
   scripts/docker-build.sh
   ```

2. Launch the web interface on `http://localhost:7860` (set `DOCKER_RUN_ARGS="--gpus all"` if you have an NVIDIA GPU exposed via the Container Toolkit):

   ```bash
   scripts/docker-run-web.sh
   ```

3. Kick off training inside a container. To enable Weights & Biases logging, store your API token in `.wandb_api_key` (or set `WANDB_KEY_FILE` to point at a different file) before running the script:

   ```bash
   scripts/docker-run-train.sh
   ```

All scripts mount the local `data`, `models`, `wandb`, and `lightning_logs` folders into the container so that checkpoints, calibration artifacts, and experiment logs persist on the host machine. Adjust the `IMAGE_TAG`, `PORT`, `CONTAINER_NAME`, or `DOCKER_RUN_ARGS` environment variables to customize the workflow.

### Usage

#### Training Models

To train the models, run:

```bash
python3 -m src.main --train
```

This will start the training process.

> **Weights & Biases logging:** place your API key in `.wandb_api_key` (or set `WANDB_API_KEY_FILE` to another path) to enable W&B tracking. If the file is missing, empty, or the key is invalid, training automatically falls back to the default Lightning logger.

#### Starting the Web Service

To start the web interface for making predictions:

```bash
python3 -m src.main
```

The web service will be available at `http://localhost:7860` by default and using the configurations specified in `static/config.json`.

## Project Structure

- `src/`: Source code for the implementation
  - `calibration/`: Conformal prediction and calibration utilities
  - `models/`: Model architectures and prediction logic
  - `data/`: Dataset handling and preprocessing
- `scripts/`: Data preparation and utility scripts
- `notebooks/`: Jupyter notebooks for analysis and experiments
- `doc/`: Detailed documentation
- `static/`: Configuration files and web assets
