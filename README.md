# Conformal prediction with task-wise calibration in a disentangled framework for multi-output image classification

This repository contains the reproducibility package for the article **Conformal prediction with task-wise calibration in a disentangled framework for multi-output image classification**. It implements the training, calibration, evaluation, and demonstration code used to compare conformal prediction strategies for two-output image classification.

The experiments study how the choice of model representation and calibration granularity affects conformal prediction sets. A low-level model treats each output combination as one joint class; a high-level model uses a shared image encoder with one classifier head per task. Connectors allow either model representation to be calibrated in the joint label space or in the task-wise label space.

## What is implemented

- **Datasets:** SGVehicle (`Color`, `Type`) and UTKFace (`Gender`, `Race`).
- **Model levels:** `LowLevelModel` for joint labels and `HighLevelModel` for task-wise labels.
- **Calibration levels:** `LowLevelCalibrator` for the joint label space and `HighLevelCalibrator` for per-task calibration.
- **Nonconformity scores:** `hinge`, `margin`, and `pip`.
- **Conformal predictors:** global standard CP, task-wise standard CP, classwise CP, global clustered CP, and task-wise clustered CP where the selected calibration level supports them.
- **Evaluation outputs:** coverage, task-wise coverage, efficiency, informativeness, and CovGap metrics.

The full experiment matrix is encoded under `experiments/{utkface,sgvehicle}/{hinge,margin,pip}/`. Each filename describes the model and calibration levels:

- `ll_ll_cal.yaml`: low-level model, low-level calibration.
- `ll_hl_cal.yaml`: low-level model, high-level calibration.
- `hl_ll_cal.yaml`: high-level model, low-level calibration.
- `hl_hl_cal.yaml`: high-level model, high-level calibration.

## Repository layout

- `src/`: implementation code.
- `src/main.py`: CLI entry point for experiments, training, calibration, and the web service.
- `src/core/`: model definitions, calibrators, connectors, predictor wrapper, and artifact IO.
- `src/calibration/`: nonconformity scores and threshold computation.
- `src/data/`: dataset and data-module logic.
- `experiments/`: YAML configurations for all dataset, score, model-level, and calibration-level combinations.
- `results/`: per-configuration CSV results and workbook summaries.
- `notebooks/`: exploratory checks and result aggregation notebooks.
- `scripts/`: data preparation and Docker helper scripts.
- `doc/`: supporting implementation notes.
- `static/`: web-service configuration and styles.

## Environment

Use Python 3.12 or newer for local runs.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All local commands below are intended to be run from the repository root with `src` on `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

Weights & Biases logging is enabled only when `.wandb_api_key` contains a valid key. Without that file, PyTorch Lightning falls back to its default local logger.

## Data preparation

Prepare both datasets and the experiment splits with:

```bash
bash scripts/prepare-data.sh
```

The script downloads SGVehicle and UTKFace archives, extracts them into `data/`, and runs the preprocessing scripts used by the experiment configs.

## Running experiments locally

The article configs are calibration/evaluation configs: by default they expect an already trained checkpoint in the configured `artifact_root`, then calibrate and evaluate each conformal-prediction variant.

Train the four base model checkpoints once:

```bash
PYTHONPATH=src python3 -m main train experiments/utkface/hinge/ll_ll_cal.yaml
PYTHONPATH=src python3 -m main train experiments/utkface/hinge/hl_hl_cal.yaml
PYTHONPATH=src python3 -m main train experiments/sgvehicle/hinge/ll_ll_cal.yaml
PYTHONPATH=src python3 -m main train experiments/sgvehicle/hinge/hl_hl_cal.yaml
```

The selected nonconformity score does not affect training, so one low-level and one high-level checkpoint per dataset can be reused across `hinge`, `margin`, and `pip` calibrations.

Run one configuration:

```bash
PYTHONPATH=src python3 -m main experiment experiments/utkface/hinge/ll_ll_cal.yaml
```

Run every YAML below a directory:

```bash
PYTHONPATH=src python3 -m main experiment_directory experiments/utkface
PYTHONPATH=src python3 -m main experiment_directory experiments/sgvehicle
```

Useful single-purpose commands:

```bash
PYTHONPATH=src python3 -m main train experiments/utkface/hinge/ll_ll_cal.yaml
PYTHONPATH=src python3 -m main calibrate experiments/utkface/hinge/ll_ll_cal.yaml
```

## Artifacts and results

Training writes checkpoints under the configured model artifact directory:

```text
artifacts/UTKFace/ll_model/models/low-model.ckpt
artifacts/UTKFace/hl_model/models/high-model.ckpt
artifacts/SGVehicle/ll_model/models/low-model.ckpt
artifacts/SGVehicle/hl_model/models/high-model.ckpt
```

Calibration writes thresholds below the same artifact root:

```text
artifacts/<dataset>/<level>_model/thresholds/<calibration-level>/<score>/<cp-type>/alpha_0.05.json
```

Experiment CSVs are written to:

```text
results/<dataset>/<score>/<config-name>-results.csv
```

The checked-in `results/*.xlsx` files provide workbook summaries for the article tables, and `notebooks/multioutput_cp_aggregation.ipynb` contains the aggregation workflow.

## Docker workflow

Build the image:

```bash
scripts/docker-build.sh
```

Prepare data in Docker:

```bash
scripts/docker-prepare-data.sh
```

Run training or calibration:

```bash
scripts/docker-run-train.sh [HOST_DATA_DIR] [HOST_ARTIFACTS_DIR] [CONFIG]
scripts/docker-run-calibrate.sh [HOST_DATA_DIR] [HOST_ARTIFACTS_DIR] [CONFIG]
```

The scripts default to `./data`, `./artifacts`, and `experiments/utkface/hinge/ll_ll_cal.yaml`. See `doc/docker-workflows.md` for environment variables such as `IMAGE_TAG`, `DOCKER_RUN_ARGS`, `RUN_AS_HOST_UID`, and `WANDB_KEY_FILE`.

## Web demo

After checkpoints and thresholds have been generated, launch the Gradio interface:

```bash
PYTHONPATH=src python3 -m main web_service
```

The UI listens on `http://localhost:7860` by default and reads `static/ws-config.json`. In Docker:

```bash
PORT=7860 scripts/docker-run-web.sh [HOST_ARTIFACTS_DIR]
```

## Additional documentation

- [Data acquisition](doc/data-acquisition.md)
- [Docker overview](doc/docker-overview.md)
- [Docker workflows](doc/docker-workflows.md)
- [Model definition](doc/model-definition.md)
- [Conformal prediction](doc/conformal-prediction.md)
- [Metrics](doc/metrics.md)
- [Web interface](doc/web-interface.md)
