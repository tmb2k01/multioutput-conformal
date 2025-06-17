# Model Definition Documentation

## Introduction

This project implements a multi-output classification system that can handle multiple classification tasks simultaneously. The system is designed to work with two main datasets:

1. **SGVehicle Dataset**: Classifies vehicles based on:
   - Color (12 classes: Yellow, Orange, Green, Gray, Red, Blue, White, Golden, Brown, Black, Purple, Pink)
   - Type (11 classes: Sedan, SUV, Van, Hatchback, MPV, Pickup, Bus, Truck, Estate, Sportscar, RV)

2. **UTKFace Dataset**: Classifies faces based on:
   - Gender (2 classes: Male, Female)
   - Race (5 classes: White, Black, Asian, Indian, Others)

## Model Architecture

The system implements two different approaches to multi-task classification:

### 1. High-Level Model

The High-Level Model uses a shared feature extractor with separate classification heads for each task. This approach allows for independent task predictions while leveraging shared features.

#### Architecture Components:
- **Feature Extractor**: 
  - Uses a pre-trained ResNet50 backbone
  - The backbone is frozen during training to preserve learned features
  - Removes the final classification layer

- **Task-Specific Heads**:
  - Separate MLP classifier for each task
  - Each head consists of:
    - Linear layer (2048 → 512)
    - ReLU activation
    - Dropout (0.5)
    - Linear layer (512 → num_classes)
  - Each head is trained independently

### 2. Low-Level Model

The Low-Level Model combines all tasks into a single joint classification problem by encoding multiple task outputs into a single class ID.

#### Architecture Components:
- **Feature Extractor**:
  - Same pre-trained ResNet50 backbone as High-Level Model
  - Backbone is frozen during training

- **Joint Classifier Head**:
  - Single MLP classifier for all tasks
  - Architecture:
    - Linear layer (2048 → 512)
    - ReLU activation
    - Dropout (0.5)
    - Linear layer (512 → total_joint_classes)
  - Uses a multiplier system to encode multiple task outputs into a single class ID

## Training Process

Both models are trained using:
- PyTorch Lightning framework
- AdamW optimizer
- Cross-entropy loss
- Early stopping with patience=5
- Maximum 30 epochs
- GPU acceleration
- WandB logging for experiment tracking
