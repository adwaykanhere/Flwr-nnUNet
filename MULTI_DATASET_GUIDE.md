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

### ✅ **Dataset Compatibility Analysis**
- Automatic compatibility checking between datasets
- Label harmonization recommendations
- Modality distribution analysis

## Quick Start

### Basic Multi-Dataset Setup

```bash
# Activate conda environment
conda activate flwrtest

# Simple 3-client setup with different datasets
python run_federated_deployment.py \
    --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}' \
    --clients 3 --rounds 5 --local-epochs 2 \
    --enable-modality-aggregation
```

### With Custom Modality Weights

```bash
# Advanced setup with custom modality weights
python run_federated_deployment.py \
    --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}' \
    --clients 3 --rounds 10 --local-epochs 3 \
    --enable-modality-aggregation \
    --modality-weights '{"CT": 0.4, "MR": 0.6}' \
    --validate --validation-frequency 2
```

### Dataset Compatibility Validation

```bash
# Validate dataset compatibility before training
python run_federated_deployment.py \
    --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}' \
    --validate-datasets
```

## Configuration Methods

### Method 1: Command Line JSON

```bash
python run_federated_deployment.py \
    --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}' \
    --enable-modality-aggregation
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
python run_federated_deployment.py --clients 3 --enable-modality-aggregation
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
python federation_config.py  # Load and validate config
python run_federated_deployment.py --config federation_config.yaml
```

## Multi-Dataset Examples

### Example 1: Multi-Modal Hospital Network

**Scenario**: 4 hospitals with different imaging capabilities

```bash
python run_federated_deployment.py --mode run \
    --client-datasets '{
        "0": "Dataset005_Prostate",
        "1": "Dataset009_Spleen", 
        "2": "Dataset027_ACDC",
        "3": "Dataset137_BraTS21"
    }' \
    --clients 4 --rounds 15 --local-epochs 4 \
    --enable-modality-aggregation \
    --modality-weights '{"CT": 0.3, "MR": 0.5, "PET": 0.2}' \
    --validate --validation-frequency 3
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
python run_federated_deployment.py --mode run \
    --client-datasets '{
        "0": "Dataset005_Prostate",
        "1": "Dataset005_Prostate",
        "2": "Dataset009_Spleen",
        "3": "Dataset027_ACDC",
        "4": "Dataset027_ACDC"
    }' \
    --clients 5 --rounds 12 --local-epochs 3 \
    --enable-modality-aggregation \
    --validate
```

### Example 3: Cross-Institutional Validation

**Scenario**: Training on Institution A data, validating on Institution B data

```bash
# Hospital Network Federation
python run_federated_deployment.py --mode run \
    --client-datasets '{
        "0": "Dataset005_Prostate",
        "1": "Dataset006_Prostate_External",
        "2": "Dataset009_Spleen",
        "3": "Dataset010_Spleen_External"
    }' \
    --clients 4 --rounds 20 --local-epochs 5 \
    --enable-modality-aggregation \
    --modality-weights '{"CT": 0.5, "MR": 0.5}' \
    --validate --validation-frequency 1 \
    --output-dir "cross_institutional_models"
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
--local-epochs 1

# Reduce validation frequency  
--validation-frequency 5

# Use CPU training for testing
export CUDA_VISIBLE_DEVICES=""
```

### Debugging Tips

#### Enable Detailed Logging

```bash
# Add debugging to environment
export PYTHONPATH="${PYTHONPATH}:."
export CUDA_LAUNCH_BLOCKING=1

# Run with verbose output
python run_federated_deployment.py --validate-datasets --client-datasets '...' --mode run
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
--clients 3 --rounds 10 --local-epochs 5 --validation-frequency 2
```

#### Medium Networks (4-6 clients)  
```bash
--clients 5 --rounds 15 --local-epochs 3 --validation-frequency 3
```

#### Large Networks (7+ clients)
```bash
--clients 10 --rounds 20 --local-epochs 2 --validation-frequency 5
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