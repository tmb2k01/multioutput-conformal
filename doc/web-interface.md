# Web Interface Documentation

## Overview

The web interface is built using Gradio and provides an interactive way to use the multi-output classification system with conformal prediction. It allows users to upload images and get predictions with uncertainty quantification using various conformal prediction approaches.

## Interface Components

### 1. Input Section
- **Image Upload**: Users can upload images through two methods:
  - Direct file upload
  - Clipboard paste
- **Model Configuration**:
  - Model Level Selection (HIGH/LOW)
  - Model Type Selection (SGVehicle/UTKFace)
  - Nonconformity Score Type (Hinge/Margin/PiP)
  - Conformal Prediction Type selection

### 2. Output Section
- **Prediction Results Table**: Displays the conformal prediction set.
  - High-level models: two columns (`Task`, `Class`).
  - Low-level models: a single column listing the joint classes (the cartesian
    product of the per-task classes).
  - A green indicator (🟢) marks each class that is included in the prediction set
    for the uploaded image.

## Configuration

The web interface is configured through a JSON file, by default
`static/ws-config.json` (override with `--config`). Trained artifacts are located
under the directory given by the `ARTIFACTS_ROOT` environment variable (default
`./artifacts/artifacts`), joined with the per-level `artifacts` subpath below.

### 1. Model Configurations
Each model configuration includes:
- **alpha**: Miscoverage level used to load the matching calibration thresholds.
- **tasks**: List of classification tasks with their classes.
- **artifacts**: Per-level artifact roots (relative to `ARTIFACTS_ROOT`). Each root
  holds the model checkpoint under `models/` and the thresholds under
  `thresholds/` (as produced by `ConformalPredictor`).
  - `low`: artifact root for the low-level model
  - `high`: artifact root for the high-level model

### Example Configuration
```json
{
  "SGVehicle": {
    "alpha": 0.05,
    "tasks": [
      {
        "name": "Color",
        "classes": ["Yellow", "Orange", "Green", "..."]
      },
      {
        "name": "Type",
        "classes": ["Sedan", "SUV", "Van", "..."]
      }
    ],
    "artifacts": {
      "low": "SGVehicle/ll_model",
      "high": "SGVehicle/hl_model"
    }
  }
}
```

## Usage

1. **Starting the Interface** (run with `PYTHONPATH=src`):
   ```bash
   python -m main web_service                       # uses static/ws-config.json
   python -m main web_service --config <path.json>  # custom config
   ```

2. **Making Predictions**:
   1. Upload an image using the image input component
   2. Select the desired model configuration:
      - Choose between HIGH or LOW level model
      - Select the model type (SGVehicle/UTKFace)
      - Choose the nonconformity score type
      - Select the conformal prediction approach
   3. Click the "Predict" button
   4. View the results in the prediction table

3. **Understanding Results**:
   - The table lists every class (per task for high-level, or joint class for
     low-level).
   - Green checkmarks (🟢) indicate the classes contained in the conformal
     prediction set for the uploaded image at the configured `alpha`.
   - If a checkpoint or threshold file is missing for the chosen combination, the
     table shows a short "Unavailable for this configuration" message instead.

## Supported Conformal Prediction Types

### High-Level Model
- Standard CP - Global Threshold
- Standard CP - Taskwise Threshold
- Classwise CP
- Clustered CP - Global Clusters
- Clustered CP - Taskwise Clusters

### Low-Level Model
- Standard CP - Global Threshold
- Classwise CP
- Clustered CP - Global Clusters

## Styling

The interface uses custom CSS styling defined in `static/styles.css` to provide:
- Centered headings
- Non-selectable table cells
- Fixed table layout
- Proper column width distribution
- Scrollable content areas
