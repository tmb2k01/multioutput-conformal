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
- **Prediction Results Table**: Displays predictions with the following columns:
  - Task/Class: The predicted class or task name
  - Score: The nonconformity score for the prediction
  - Threshold: The conformal prediction threshold
  - Visual indicators (🟢) for predictions that pass the threshold

## Configuration

The web interface is configured through a JSON configuration file located at `static/config.json`. The configuration file defines:

### 1. Model Configurations
Each model configuration includes:
- **Tasks**: List of classification tasks with their classes
- **Weights**: Paths to model checkpoint files
  - Low-level model weights
  - High-level model weights
- **Thresholds**: Paths to calibration files
  - Low-level model thresholds
  - High-level model thresholds

### Example Configuration
```json
{
  "SGVehicle": {
    "tasks": [
      {
        "name": "Color",
        "classes": ["Yellow", "Orange", "Green", ...]
      },
      {
        "name": "Type",
        "classes": ["Sedan", "SUV", "Van", ...]
      }
    ],
    "weights": {
      "low": "./models/sgvehicle-low-level-model.ckpt",
      "high": "./models/sgvehicle-high-level-model.ckpt"
    },
    "thresholds": {
      "low": "./models/sgvehicle-low-level-calibration.json",
      "high": "./models/sgvehicle-high-level-calibration.json"
    }
  }
}
```

## Usage

1. **Starting the Interface**:
   ```bash
   python src/main.py
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
   - The table shows predictions for each class/task
   - Nonconformity scores indicate prediction uncertainty
   - Thresholds show the conformal prediction boundaries
   - Green checkmarks (🟢) indicate predictions that pass the threshold

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
