# Evaluation Metrics

This document describes the various metrics used to evaluate the performance of our approaches, including both traditional model metrics and conformal prediction-specific metrics.

## Traditional Model Metrics

### Low-Level Accuracy
- Measures the accuracy of individual task predictions
- Calculated as the proportion of correct predictions for each task independently
- Useful for evaluating the base performance of the model before conformal prediction

### High-Level Accuracy
- Measures the overall accuracy across all tasks
- Calculated as the proportion of samples where all tasks are predicted correctly
- Provides a more stringent evaluation of model performance
- Particularly relevant for multi-task scenarios where all tasks must be correct

## Conformal Prediction Metrics

### Coverage
- Measures the proportion of prediction sets that contain the true label
- Should be close to the desired coverage level (1 - α)
- Can be computed:
  - Per task (taskwise coverage)
  - Across all tasks (overall coverage)
- Higher coverage indicates better reliability of the prediction sets

### Efficiency
- Measures the average size of the prediction sets
- Lower values indicate more precise predictions
- Can be computed:
  - Per task (taskwise efficiency)
  - Across all tasks (overall efficiency)
- Formula: Average number of predicted labels per sample

### Informativeness
- Measures the proportion of singleton prediction sets
- Higher values indicate more decisive predictions
- Can be computed:
  - Per task (taskwise informativeness)
  - Across all tasks (overall informativeness)
- Formula: Fraction of predictions that contain exactly one label

### Weighted Metrics

#### Weighted Efficiency
- Adjusts efficiency based on task importance or complexity
- Weights can be based on:
  - Number of classes per task (tasks with fewer classes get lower weights)
  - Task difficulty or importance
  - Data distribution
- Formula: Weighted average of taskwise efficiencies

#### Weighted Informativeness
- Adjusts informativeness based on task characteristics
- Weights can be based on:
  - Number of classes per task
  - Task complexity
  - Prediction confidence
- Formula: Weighted average of taskwise informativeness

## Implementation Details

The metrics are implemented in the following ways:

1. **Taskwise Metrics**
   - Computed independently for each task
   - Useful for analyzing performance variations across tasks
   - Helps identify which tasks are more challenging

2. **Overall Metrics**
   - Aggregated across all tasks
   - Provides a single performance measure
   - Useful for comparing different approaches

3. **Weighted Metrics**
   - Implemented using task-specific weights
   - Weights can be customized based on requirements
   - Helps balance the importance of different tasks
