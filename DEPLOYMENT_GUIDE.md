# Flower SuperNode/SuperLink Deployment Guide

This guide explains how to run the federated nnUNet training using Flower's native deployment engines (SuperNode and SuperLink) with modality-aware federated averaging.

## Overview

The deployment system consists of:
- **SuperLink**: Central server that coordinates federated learning
- **SuperNodes**: Client nodes that perform local training
- **Modality-aware Aggregation**: Groups clients by imaging modality (CT, MR, etc.) for improved aggregation
- **Multi-Dataset Federation**: Supports clients with different datasets for real-world hospital networks

📚 **For multi-dataset scenarios, see the comprehensive [Multi-Dataset Federation Guide](MULTI_DATASET_GUIDE.md)**

## Quick Start

#### Step 1: Set Environment Variables for Model Saving

```bash
# Set model saving configuration (REQUIRED for saving models)
export OUTPUT_ROOT="./federated_models"         # User-writable directory (recommended)
export VALIDATE_MODELS=true                     # Enable validation for model saving
```

#### Step 2: Start SuperLink (Server)

```bash
# Terminal 1: Start the SuperLink server
flower-superlink --insecure
```

#### Step 3: Start SuperNodes (Clients)

**Basic usage (uses environment variables for dataset):**
```bash
# Terminal 2: Start first SuperNode (Client 0)
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config "partition-id=0"

# Terminal 3: Start second SuperNode (Client 1)  
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config "partition-id=1"
```

**Enhanced usage with dataset and fold specification:**
```bash
# Terminal 2: Start first SuperNode with specific dataset and fold
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'

# Terminal 3: Start second SuperNode with different fold
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset005_Prostate" fold=1'

# Terminal 4: Start third SuperNode with full dataset path
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-path="/path/to/nnUNet_preprocessed/Dataset009_Spleen" fold=2'
```

#### Step 4: Run Federation

```bash
# Terminal 4: Start the federated learning
flwr run . deployment
```

## Configuration Options

### Basic Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `Dataset005_Prostate` | nnUNet dataset to use (single-dataset mode) |
| `--clients` | `2` | Number of federated clients |
| `--rounds` | `3` | Number of training rounds |
| `--local-epochs` | `1` | Local epochs per client per round |

### Multi-Dataset Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--client-datasets` | `None` | JSON mapping of client IDs to datasets |
| `--validate-datasets` | `False` | Validate dataset compatibility |

### Deployment Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `run` | Mode: `superlink`, `supernode`, or `run` |
| `--superlink-host` | `127.0.0.1` | SuperLink server address |
| `--superlink-port` | `9092` | SuperLink server port |
| `--node-id` | `0` | SuperNode identifier |
| `--partition-id` | `0` | Client ID |
| `--insecure` | `True` | Use insecure connection (for testing) |

### Model Saving Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OUTPUT_ROOT` | `./federated_models` | Directory where models are saved |
| `VALIDATE_MODELS` | `false` | Enable validation (required for model saving) |

### Modality-Aware Aggregation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-modality-aggregation` | `False` | Enable modality-aware aggregation |
| `--modality-weights` | `None` | JSON string of modality weights |

### Example Modality Weights

```bash
# Equal weighting between CT and MR
--modality-weights '{"CT": 0.5, "MR": 0.5}'

# CT-dominant weighting
--modality-weights '{"CT": 0.7, "MR": 0.3}'

# Multi-modality setup
--modality-weights '{"CT": 0.4, "MR": 0.4, "PET": 0.2}'
```

## Environment Variables

Set these environment variables for consistent configuration:

```bash
export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
export TASK_NAME="Dataset005_Prostate"
export OUTPUT_ROOT="/path/to/federated_models"  # REQUIRED for model saving
export NUM_CLIENTS=2
export NUM_TRAINING_ROUNDS=3
export LOCAL_EPOCHS=2
export ENABLE_MODALITY_AGGREGATION=true
export MODALITY_WEIGHTS='{"CT": 0.6, "MR": 0.4}'
export VALIDATE_MODELS=true                     # Enable validation for model saving
```

## Node Configuration Parameters

The `--node-config` parameter allows you to specify dataset and training configuration directly in the SuperNode command, providing maximum flexibility for manual deployments.

### Supported Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `partition-id` | int | **Required**. Unique client identifier | `partition-id=0` |
| `dataset-name` | string | Dataset name (overrides environment variables) | `dataset-name="Dataset005_Prostate"` |
| `dataset-path` | string | Full path to dataset (highest priority) | `dataset-path="/path/to/nnUNet_preprocessed/Dataset005_Prostate"` |
| `fold` | int | nnUNet cross-validation fold (0-4, default: 0) | `fold=1` |

### Parameter Priority Order

1. **`dataset-path`** - Full path specification (highest priority)
2. **`dataset-name`** - Dataset name from node-config
3. **Environment variables** - `CLIENT_DATASETS`, `CLIENT_{id}_DATASET`, `TASK_NAME`
4. **Default fallback** - `Dataset005_Prostate`

## Modality Detection

The system automatically detects client modalities from:

1. **Channel Names** in `dataset.json`:
   - CT: Contains "ct", "computed"
   - MR: Contains "mr", "magnetic", "t1", "t2"
   - PET: Contains "pet"
   - US: Contains "us", "ultrasound"

2. **Dataset Fingerprint** intensity properties

3. **Manual Override** via metadata

## Federation Phases

The federated learning process consists of:

1. **Preprocessing Round** (`-2`): Clients share fingerprints and modality info
2. **Initialization Round** (`-1`): Global model initialization
3. **Training Rounds** (`0+`): Federated training with modality-aware aggregation

## Aggregation Strategy

### Traditional FedAvg
- Weighted average by number of training examples
- All clients treated equally regardless of modality

### Modality-Aware Aggregation
1. **Intra-modality aggregation**: CT clients → CT model, MR clients → MR model
2. **Inter-modality aggregation**: Weighted combination of modality models
3. **Global model**: Final aggregated model across all modalities

## Output and Model Saving

### Model Output Structure

Models are saved to the directory specified by `OUTPUT_ROOT`:

```
/path/to/federated_models/
├── client_0/
│   └── model_best.pt              # Best model from client 0
├── client_1/
│   └── model_best.pt              # Best model from client 1
├── client_2/
│   └── model_best.pt              # Best model from client 2 (if applicable)
└── global_models/
    ├── global_best_model_modality_aware.pt  # Global aggregated model
    └── round_X_metadata.json               # Training metadata
```

### Model Saving Requirements

Models are only saved when the following conditions are met:
1. `OUTPUT_ROOT` environment variable is set
2. Validation is enabled (models save when validation improves)
3. Client achieves better validation performance than previous rounds

## Next Steps

1. Experiment with different modality weight configurations
2. Add more imaging modalities (PET, US, etc.)
3. Implement domain adaptation techniques
4. Scale to larger numbers of clients
