# Conformal Prediction Approaches

Conformal prediction is a framework for uncertainty quantification in machine learning that provides valid prediction sets with guaranteed coverage. There are five main approaches to conformal prediction, but only three are typically applicable to low-level models. This document outlines these approaches and their implementation details.

## Applicable Approaches for Low-Level Models

### 1. Standard Conformal Prediction (Global Threshold)

This is the most basic approach where a single threshold is computed for all predictions.

**Implementation Steps:**
1. Split the data into training and calibration sets
2. Train the model on the training set
3. Calculate nonconformity scores on the calibration set
4. Compute a single global threshold using the calibration scores
5. Make predictions with the global threshold

**Threshold Calculation:**
```python
def calculate_global_threshold(scores, alpha):
    # scores: nonconformity scores from calibration set
    # alpha: desired significance level (e.g., 0.1 for 90% coverage)
    threshold = np.quantile(scores, 1 - alpha)
    return threshold
```

### 2. Classwise Conformal Prediction

This approach computes separate thresholds for each class, providing more precise control over prediction sets.

**Implementation Steps:**
1. Split data into training and calibration sets
2. Train the model on the training set
3. Calculate nonconformity scores for each class on the calibration set
4. Compute separate thresholds for each class
5. Make predictions using class-specific thresholds

**Threshold Calculation:**
```python
def calculate_classwise_thresholds(scores_per_class, alpha):
    # scores_per_class: dictionary of scores for each class
    thresholds = {}
    for class_label, scores in scores_per_class.items():
        thresholds[class_label] = np.quantile(scores, 1 - alpha)
    return thresholds
```

### 3. Clustered Conformal Prediction (Global Clustering)

This approach uses clustering to group similar instances and compute thresholds for each cluster.

**Implementation Steps:**
1. Split data into training and calibration sets
2. Train the model on the training set
3. Perform clustering on the calibration set
4. Calculate nonconformity scores for each cluster
5. Compute thresholds for each cluster
6. Make predictions using cluster-specific thresholds

**Threshold Calculation:**
```python
def calculate_cluster_thresholds(cluster_scores, alpha):
    # cluster_scores: dictionary of scores for each cluster
    thresholds = {}
    for cluster_id, scores in cluster_scores.items():
        thresholds[cluster_id] = np.quantile(scores, 1 - alpha)
    return thresholds
```

## High-Level Approaches (Not Applicable to Low-Level Models)

### 1. Standard Conformal Prediction (Taskwise Threshold)
- Computes separate thresholds for different tasks
- Requires high-level model architecture
- More complex implementation
- Better suited for multi-task learning scenarios

### 2. Clustered Conformal Prediction (Taskwise Clustering)
- Combines task-specific clustering with conformal prediction
- Requires high-level model architecture
- Most complex implementation
- Best for complex multi-task scenarios

## Making Predictions

For the low-level approaches, predictions are made using the following general process:

1. Calculate the nonconformity score for the new instance
2. For classwise/clustered approaches, determine the appropriate class/cluster
3. Compare the score against the corresponding threshold
4. Include the prediction in the prediction set if the score is below the threshold

**Example Prediction Function:**
```python
def make_prediction(model, new_instance, thresholds, approach='global'):
    # Calculate nonconformity score
    score = calculate_nonconformity_score(model, new_instance)
    
    if approach == 'global':
        threshold = thresholds  # Single threshold
    elif approach == 'classwise':
        class_label = predict_class(model, new_instance)
        threshold = thresholds[class_label]
    elif approach == 'clustered':
        cluster_id = assign_cluster(new_instance)
        threshold = thresholds[cluster_id]
    
    # Make prediction
    if score <= threshold:
        return True  # Include in prediction set
    return False  # Exclude from prediction set
```

## Nonconformity Scores

The choice of nonconformity score function is crucial and depends on the model type:

1. For regression: Absolute error or squared error
2. For classification: Probability-based scores or distance-based scores
3. For anomaly detection: Reconstruction error or density estimates

## Important Considerations

1. **Coverage Guarantee**: All approaches provide valid coverage guarantees under the exchangeability assumption
2. **Efficiency**: 
   - Global threshold: Simplest but least efficient
   - Classwise: More efficient for classification tasks
   - Clustered: Most efficient but requires good clustering
3. **Computational Cost**: 
   - Global threshold: Lowest
   - Classwise: Moderate
   - Clustered: Highest
4. **Data Requirements**: All approaches require sufficient calibration data for reliable threshold estimation

## Limitations

1. The exchangeability assumption must hold for the coverage guarantees to be valid
2. The quality of prediction sets depends on:
   - Choice of nonconformity score function
   - Quality of clustering (for clustered approach)
   - Class distribution (for classwise approach)
3. The approaches may be conservative in practice, leading to larger prediction sets than necessary
4. Clustered approach requires careful selection of clustering algorithm and parameters
