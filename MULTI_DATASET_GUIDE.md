# Multi-Dataset Federation Guide

This guide explains how to use the multi-dataset federated learning capabilities of the Flower nnUNet implementation, allowing different clients to train on different datasets while maintaining intelligent aggregation strategies.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Configuration Methods](#configuration-methods)
5. [Multi-Dataset Examples](#multi-dataset-examples)
6. [Advanced Aggregation Strategies](#advanced-aggregation-strategies)
7. [Dataset Compatibility](#dataset-compatibility)
8. [Troubleshooting](#troubleshooting)

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

## Key Features

### ✅ **Automatic Dataset Detection**
- Clients automatically detect their assigned datasets
- Support for both JSON configuration and environment variables
- Validation of dataset availability and compatibility

### ✅ **Intelligent Modality Grouping**
- Automatic extraction of modality information from dataset.json
- Grouping by both dataset and modality (e.g., `Dataset005_Prostate_MR`)
- Support for CT, MR, PET, US, and custom modalities

### ✅ **Advanced Aggregation Strategies**
- **Traditional FedAvg**: Single dataset, single modality
- **Modality-Aware**: Single dataset, multiple modalities
- **Multi-Dataset**: Multiple datasets with cross-dataset modality aggregation
- **Backbone Aggregation Strategy**: Shares only middle layers while keeping first/last layers local

### ✅ **Dataset Compatibility Analysis**
- Automatic compatibility checking between datasets
- Label harmonization recommendations
- Modality distribution analysis

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

# Backbone aggregation strategy (always enabled)
# First and last layers are trained locally with 10-epoch warmup
# Only middle layers are shared for global aggregation
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

### Method 3: YAML Configuration File

Create `federation_config.yaml`:

```yaml
name: "multi_modal_medical_federation"
description: "Multi-dataset federation across different medical imaging modalities"
preprocessed_root: "/path/to/nnUNet_preprocessed"

clients:
  - client_id: "0"
    dataset: "Dataset005_Prostate"
    partition_id: 0
    local_epochs: 3
    validation_enabled: true
    gpu_id: 0
  - client_id: "1"
    dataset: "Dataset009_Spleen"
    partition_id: 1
    local_epochs: 3
    validation_enabled: true
    gpu_id: 0
  - client_id: "2"
    dataset: "Dataset002_Heart"
    partition_id: 2
    local_epochs: 3
    validation_enabled: true
    gpu_id: 0

datasets:
  - name: "Dataset005_Prostate"
    path: "/path/to/nnUNet_preprocessed/Dataset005_Prostate"
    modality: "MR"
    description: "Prostate segmentation (MR T2-weighted)"
    priority: 1.0
  - name: "Dataset009_Spleen"
    path: "/path/to/nnUNet_preprocessed/Dataset009_Spleen"
    modality: "CT"
    description: "Spleen segmentation (CT)"
    priority: 1.0
  - name: "Dataset002_Heart"
    path: "/path/to/nnUNet_preprocessed/Dataset002_Heart"
    modality: "MR"
    description: "Heart segmentation (MR cine)"
    priority: 1.0

aggregation:
  strategy: "multi_dataset"
  enable_modality_aggregation: true
  modality_weights:
    CT: 0.4
    MR: 0.6
  dataset_modality_weights:
    "Dataset005_Prostate_MR": 0.3
    "Dataset009_Spleen_CT": 0.4
    "Dataset027_ACDC_MR": 0.3

training:
  rounds: 10
  local_epochs: 3
  validation_enabled: true
  validation_frequency: 2
  save_frequency: 2
  output_dir: "multi_dataset_models"
```

Run with configuration file:

```bash
# Load configuration into environment
python federation_config.py  # Load and validate config

# Start manual deployment with config-based environment variables
flower-superlink --insecure  # Terminal 1
# Start SuperNodes as configured in YAML
flwr run . deployment  # After all SuperNodes are running
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

**Expected Output**:
```
📊 Federation Type: Multi-Dataset (4 datasets)
🏥 Client-Dataset Mapping:
   Client 0: Dataset005_Prostate
   Client 1: Dataset009_Spleen
   Client 2: Dataset027_ACDC
   Client 3: Dataset137_BraTS21
🧠 Detected modalities: ['CT', 'MR']
🔄 Multi-dataset federation: 4 datasets across 4 clients
🧠 Automatically enabling modality-aware aggregation for multi-dataset federation
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

1. **Backbone Aggregation** (always enabled)
   - Clients warm up first/last layers locally for 10 epochs in round 0
   - Only middle layers (backbone) are shared and aggregated globally
   - First and last layers remain client-specific for data heterogeneity
   - Best for heterogeneous datasets with different input/output structures

2. **Modality-Aware Backbone Aggregation** (`modality_aggregation=true`)
   - Combines backbone aggregation with modality-aware grouping
   - Aggregates backbone within modalities, then across modalities
   - Best for multi-modal heterogeneous setups

3. **Multi-Dataset Backbone Aggregation** (`num_datasets>1`)
   - Backbone aggregation applied across multiple datasets
   - Handles different datasets with varying architectures seamlessly
   - Best for real-world federated medical imaging scenarios

### Backbone Aggregation Warmup Process

The backbone aggregation strategy includes a crucial warmup phase:

#### Round 0: Warmup Phase
1. **Local Training**: Each client trains first/last layers locally for 10 epochs
2. **No Global Sharing**: First and last layer weights are never shared with server
3. **Warmup Flag**: Clients send `is_warmup: true` to indicate warmup status
4. **Backbone Collection**: Server collects backbone parameters but doesn't aggregate yet

#### Round 1+: Global Aggregation
1. **Backbone Sharing**: Only middle layer parameters are sent to server
2. **Global Aggregation**: Server averages backbone parameters across clients
3. **Local Preservation**: First/last layers remain client-specific throughout training
4. **Continued Training**: Clients continue local training with updated global backbone

#### Benefits of Warmup Strategy
- **Data Heterogeneity**: Handles different input channels and output classes seamlessly
- **Architecture Flexibility**: No need for manual architecture harmonization
- **Performance**: Maintains local adaptation while gaining from global knowledge
- **Simplicity**: Eliminates complex parameter matching and harmonization

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

#### Example Compatibility Report

```
📋 Compatibility Analysis:
🧠 Detected modalities: ['CT', 'MR']
🏷️  Modality distribution: {'CT': 2, 'MR': 3}

💡 Recommendations:
✅ Multi-modality datasets detected: ['CT', 'MR'] - use modality-aware aggregation
⚠️  Label conflicts detected: ['1', '2'] - harmonization recommended
✅ Common labels found: ['0'] - direct aggregation possible
💡 Large number of datasets - consider hierarchical aggregation
```

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

## Troubleshooting

### Common Issues

#### 1. Dataset Not Found

```
❌ Dataset 'Dataset009_Spleen' for client 1 is invalid or missing
```

**Solution**:
```bash
# Check if dataset exists
ls /path/to/nnUNet_preprocessed/Dataset009_Spleen/

# Verify required files
ls /path/to/nnUNet_preprocessed/Dataset009_Spleen/nnUNetPlans.json
ls /path/to/nnUNet_preprocessed/Dataset009_Spleen/dataset.json
```

#### 2. Modality Detection Issues

```
⚠️  Client modality detection failed - using UNKNOWN
```

**Solution**:
- Check `dataset.json` channel_names format:
```json
{
  "channel_names": {
    "0": "T2w"  // Should contain modality keywords
  }
}
```

#### 3. Aggregation Failures

```
❌ No valid dataset-modality groups found
```

**Solution**:
- Ensure clients have transmitted metadata during preprocessing phase
- Check server logs for client registration
- Verify modality extraction is working

#### 4. Memory Issues with Large Multi-Dataset Setups

**Solution**:
```bash
# Reduce local epochs
export LOCAL_EPOCHS=1

# Reduce validation frequency  
export VALIDATION_FREQUENCY=5

# Use CPU training for testing
export CUDA_VISIBLE_DEVICES=""
```

#### 5. Models Not Being Saved in Multi-Dataset Setup

```
❌ Models not found in output directory after training
```

**Solution**:
```bash
# Check if OUTPUT_ROOT is set
echo $OUTPUT_ROOT

# Set OUTPUT_ROOT if missing
export OUTPUT_ROOT="./multi_dataset_models"

# Ensure validation is enabled (required for model saving)
export VALIDATE_MODELS=true

# Check directory structure after training
ls -la $OUTPUT_ROOT/client_*/model_best.pt

# Expected multi-dataset model structure:
# /path/to/multi_dataset_models/
# ├── client_0/  (Dataset005_Prostate)
# │   └── model_best.pt
# ├── client_1/  (Dataset009_Spleen)  
# │   └── model_best.pt
# ├── client_2/  (Dataset002_Heart)
# │   └── model_best.pt
# └── global_models/
#     └── global_best_model_modality_aware.pt
```

### Debugging Tips

#### Enable Detailed Logging

```bash
# Set model saving for debugging
export OUTPUT_ROOT="/path/to/debug_models"
export VALIDATE_MODELS=true

# Add debugging to environment
export PYTHONPATH="${PYTHONPATH}:."
export CUDA_LAUNCH_BLOCKING=1
export CLIENT_DATASETS='{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}'

# Run with manual deployment and verbose output
flower-superlink --insecure --verbose  # Terminal 1

# Start SuperNodes with debugging
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate"' --verbose

flwr run . deployment --verbose  # After SuperNodes are running
```

#### Validate Configuration

```python
# Test configuration before deployment
from federation_config import FederationConfigManager

manager = FederationConfigManager()
config = manager.create_config_from_args(args, preproc_root)
validation_report = manager.validate_config(config)

if not validation_report["valid"]:
    print("Configuration errors:", validation_report["errors"])
```

#### Check Dataset Compatibility

```python
# Analyze dataset compatibility
from dataset_compatibility import create_multi_dataset_config

client_datasets = {"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}
config = create_multi_dataset_config(client_datasets, preproc_root)

print("Compatibility:", config["compatibility_analysis"])
print("Recommendations:", config["compatibility_analysis"]["recommendations"])
```

## Performance Optimization

### Recommended Settings

#### Small Networks (2-3 clients)
```bash
# Configure environment for small network
export NUM_ROUNDS=10
export LOCAL_EPOCHS=5
export VALIDATION_FREQUENCY=2
# Start 3 SuperNodes with different datasets
```

#### Medium Networks (4-6 clients)  
```bash
# Configure environment for medium network
export NUM_ROUNDS=15
export LOCAL_EPOCHS=3
export VALIDATION_FREQUENCY=3
# Start 5 SuperNodes with appropriate dataset distribution
```

#### Large Networks (7+ clients)
```bash
# Configure environment for large network
export NUM_ROUNDS=20
export LOCAL_EPOCHS=2
export VALIDATION_FREQUENCY=5
# Start 10+ SuperNodes with hierarchical dataset organization
```

### Multi-Dataset Specific Optimizations

1. **Hierarchical Aggregation**: For 5+ datasets, consider grouping by anatomical region
2. **Weighted Validation**: Use dataset-specific validation frequencies
3. **Adaptive Learning Rates**: Different rates for different modalities
4. **Early Stopping**: Stop clients with converged models early

## Next Steps

1. **Experiment with Weight Configurations**: Try different modality and dataset weights
2. **Cross-Validation**: Validate models on held-out datasets from other institutions
3. **Continual Learning**: Add new datasets to existing federations
4. **Domain Adaptation**: Implement domain-specific fine-tuning strategies

For more advanced scenarios, see the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment considerations.