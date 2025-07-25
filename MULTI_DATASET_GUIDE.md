# Multi-Dataset Federation Guide

This guide explains how to use the multi-dataset federated learning capabilities of the Flower nnUNet implementation, allowing different clients to train on different datasets while maintaining intelligent aggregation strategies.

## Table of Contents

1. [Overview](#overview)
3. [Quick Start](#quick-start)
4. [Configuration Methods](#configuration-methods)
5. [Multi-Dataset Examples](#multi-dataset-examples)
6. [Advanced Aggregation Strategies](#advanced-aggregation-strategies)
7. [Dataset Compatibility](#dataset-compatibility)

## Overview

Multi-dataset federation enables federated learning scenarios where:

- **Different institutions** have different medical datasets
- **Different modalities** are distributed across clients (CT, MR, PET, US)
- **Cross-dataset generalization** is needed for robust model performance
- **Heterogeneous data distributions** are common in real-world deployments

### Traditional vs Multi-Dataset Federation

| Aspect | Traditional Federation | Multi-Dataset Federation |
|--------|----------------------|--------------------------|
| **Data Distribution** | Same dataset across all clients | Different datasets per client |
| **Modality Handling** | Single modality assumption | Multi-modal aware aggregation |
| **Aggregation Strategy** | Simple weighted averaging | Dataset-modality group aggregation |
| **Real-world Applicability** | Limited to identical setups | Supports diverse hospital networks |


## Quick Start

### Manual Multi-Dataset Federation Deployment

#### Step 1: Set Environment Variables

```bash
# Activate conda environment
conda activate flwrtest

# Set model saving configuration (REQUIRED)
export OUTPUT_ROOT="./multi_dataset_models"        # User-writable directory
export VALIDATE_MODELS=true                        # Enable validation for model saving

# Set multi-dataset configuration
export CLIENT_DATASETS='{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}'
export ENABLE_MODALITY_AGGREGATION=true
export MODALITY_WEIGHTS='{"CT": 0.4, "MR": 0.6}'

# Enable FedMA architecture harmonization (default: enabled)
export ENABLE_FEDMA=true
export FEDMA_CHANNEL_HARMONIZATION=true  # Handle different input channels
export FEDMA_CLASS_HARMONIZATION=true    # Handle different output classes
export FEDMA_LAYER_MATCHING=true         # Enable neuron matching
```

#### Step 2: Start SuperLink (Server)

```bash
# Terminal 1: Start the SuperLink server
flower-superlink --insecure
```

#### Step 3: Start SuperNodes (Clients) with Different Datasets

```bash
# Terminal 2: Start first SuperNode with Prostate dataset
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'

# Terminal 3: Start second SuperNode with Spleen dataset  
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset009_Spleen" fold=1'

# Terminal 4: Start third SuperNode with Heart dataset
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-name="Dataset002_Heart" fold=2'
```

#### Step 4: Run Federation

```bash
# Terminal 5: Start the federated learning
flwr run . deployment
```

## Configuration Methods

### Method 1: Environment Variables with Node Config

```bash
# Set model saving configuration (REQUIRED)
export OUTPUT_ROOT="/path/to/multi_dataset_models"
export VALIDATE_MODELS=true

# Set multi-dataset configuration
export CLIENT_DATASETS='{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}'
export ENABLE_MODALITY_AGGREGATION=true

# Start SuperNodes with specific datasets
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate"'
```

### Method 2: Environment Variables

```bash
# Set client-dataset mapping
export CLIENT_DATASETS='{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}'

# Set individual client datasets (backup method)
export CLIENT_0_DATASET="Dataset005_Prostate"
export CLIENT_1_DATASET="Dataset009_Spleen"
export CLIENT_2_DATASET="Dataset002_Heart"

# Run federation
flwr run . deployment
```

## Multi-Dataset Examples

### Example 1: Multi-Modal Hospital Network

**Scenario**: 4 hospitals with different imaging capabilities

```bash
# Set model saving configuration
export OUTPUT_ROOT="/path/to/multi_modal_hospital_models"
export VALIDATE_MODELS=true

# Set environment variables
export CLIENT_DATASETS='{
    "0": "Dataset005_Prostate",
    "1": "Dataset009_Spleen", 
    "2": "Dataset027_ACDC",
    "3": "Dataset137_BraTS21"
}'
export ENABLE_MODALITY_AGGREGATION=true
export MODALITY_WEIGHTS='{"CT": 0.3, "MR": 0.5, "PET": 0.2}'

# Terminal 1: Start SuperLink
flower-superlink --insecure

# Terminal 2-5: Start SuperNodes
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset009_Spleen" fold=1'

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-name="Dataset027_ACDC" fold=2'

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9097 \
    --node-config 'partition-id=3 dataset-name="Dataset137_BraTS21" fold=3'

# Terminal 6: Run federation
flwr run . deployment
```

### Example 2: Radiology Department Specialization

**Scenario**: Different departments specializing in different body parts

```bash
# Set model saving configuration
export OUTPUT_ROOT="/path/to/radiology_dept_models"
export VALIDATE_MODELS=true

# Set environment variables for department specialization
export CLIENT_DATASETS='{
    "0": "Dataset005_Prostate",
    "1": "Dataset005_Prostate",
    "2": "Dataset009_Spleen",
    "3": "Dataset027_ACDC",
    "4": "Dataset027_ACDC"
}'
export ENABLE_MODALITY_AGGREGATION=true

# Start SuperLink
flower-superlink --insecure  # Terminal 1

# Start SuperNodes for specialized departments
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'  # Prostate Dept A

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset005_Prostate" fold=1'  # Prostate Dept B

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-name="Dataset009_Spleen" fold=2'    # Spleen Dept

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9097 \
    --node-config 'partition-id=3 dataset-name="Dataset027_ACDC" fold=3'     # Cardiology A

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9098 \
    --node-config 'partition-id=4 dataset-name="Dataset027_ACDC" fold=4'     # Cardiology B

# Run federation
flwr run . deployment  # Terminal 7
```

### Example 3: Cross-Institutional Validation

**Scenario**: Training on Institution A data, validating on Institution B data

```bash
# Set model saving configuration
export OUTPUT_ROOT="/path/to/cross_institutional_models"
export VALIDATE_MODELS=true

# Set cross-institutional environment
export CLIENT_DATASETS='{
    "0": "Dataset005_Prostate",
    "1": "Dataset006_Prostate_External",
    "2": "Dataset009_Spleen",
    "3": "Dataset010_Spleen_External"
}'
export ENABLE_MODALITY_AGGREGATION=true
export MODALITY_WEIGHTS='{"CT": 0.5, "MR": 0.5}'

# Start SuperLink
flower-superlink --insecure  # Terminal 1

# Cross-institutional SuperNodes
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'        # Institution A - Prostate

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset006_Prostate_External" fold=1' # Institution B - Prostate

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-name="Dataset009_Spleen" fold=2'          # Institution A - Spleen

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9097 \
    --node-config 'partition-id=3 dataset-name="Dataset010_Spleen_External" fold=3'  # Institution B - Spleen

# Run cross-institutional federation
flwr run . deployment  # Terminal 6
```

## Advanced Aggregation Strategies

### Strategy Selection

The system automatically selects the best aggregation strategy:

1. **Traditional FedAvg** (`num_datasets=1, modality_aggregation=false`)
   - Simple weighted averaging by number of examples
   - Best for homogeneous setups

2. **Modality-Aware FedAvg** (`num_datasets=1, modality_aggregation=true`)
   - Aggregates within modalities, then across modalities
   - Best for single dataset with multiple modalities

3. **Multi-Dataset Modality-Aware** (`num_datasets>1, modality_aggregation=true`)
   - Aggregates within dataset-modality groups, then across groups
   - Best for heterogeneous multi-dataset setups

### Custom Aggregation Weights

#### Modality-Based Weights

```json
{
  "CT": 0.4,
  "MR": 0.5,
  "PET": 0.1
}
```

#### Dataset-Modality Specific Weights

```json
{
  "Dataset005_Prostate_MR": 0.25,
  "Dataset009_Spleen_CT": 0.35,
  "Dataset027_ACDC_MR": 0.25,
  "Dataset137_BraTS21_MR": 0.15
}
```

#### Priority-Based Weights

```json
{
  "high_quality_data": 0.6,
  "standard_data": 0.3,
  "limited_data": 0.1
}
```

## Dataset Compatibility

### Automatic Compatibility Analysis

The system analyzes compatibility across multiple dimensions:

#### Modality Compatibility
- **CT + CT**: High compatibility (1.0)
- **MR + MR**: High compatibility (1.0)  
- **CT + MR**: Medium compatibility (0.6)
- **PET + US**: Low compatibility (0.3)

#### Label Compatibility
- **Common Labels**: Direct aggregation possible
- **Overlapping Labels**: Partial compatibility
- **Disjoint Labels**: Requires harmonization


### Label Harmonization

When datasets have different label mappings:

```python
# Automatic label harmonization
from dataset_compatibility import DatasetCompatibilityManager

manager = DatasetCompatibilityManager()
harmonization_map = manager.harmonize_labels(strategy="intersection")
# Uses only common labels across all datasets

harmonization_map = manager.harmonize_labels(strategy="union") 
# Maps all labels to a common space
```

## Next Steps

1. **Experiment with Weight Configurations**: Try different modality and dataset weights
2. **Cross-Validation**: Validate models on held-out datasets from other institutions
3. **Continual Learning**: Add new datasets to existing federations
4. **Domain Adaptation**: Implement domain-specific fine-tuning strategies

For more advanced scenarios, see the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment considerations.
