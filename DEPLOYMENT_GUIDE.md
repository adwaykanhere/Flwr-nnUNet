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
export OUTPUT_ROOT="/path/to/federated_models"  # Where to save trained models
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
| `OUTPUT_ROOT` | `/local/pathto/nnunet_output` | Directory where models are saved |
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

### Cross-Validation Folds Explained

nnUNet uses **5-fold cross-validation** where each fold represents a different train/validation split:

- **Fold 0**: ~80% train, ~20% validation (cases vary by split)
- **Fold 1**: Different 80/20 split with different validation cases
- **Fold 2-4**: Additional splits ensuring all data is used for validation

**Benefits for Federated Learning:**
- **Same Dataset, Different Folds**: Simulates different patient populations per institution
- **Data Diversity**: Each client trains on different subsets, improving federation robustness
- **Real-world Simulation**: Mimics how different hospitals have different patient cohorts

### Usage Examples

**Multi-Institution Simulation (Same Dataset, Different Folds):**
```bash
# Hospital A: Prostate dataset, fold 0
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'

# Hospital B: Prostate dataset, fold 1  
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset005_Prostate" fold=1'

# Hospital C: Prostate dataset, fold 2
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-name="Dataset005_Prostate" fold=2'
```

**True Multi-Institution (Different Datasets):**
```bash
# Hospital A: Prostate dataset
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'partition-id=0 dataset-name="Dataset005_Prostate" fold=0'

# Hospital B: Spleen dataset
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'partition-id=1 dataset-name="Dataset009_Spleen" fold=1'

# Hospital C: Heart dataset with full path
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9096 \
    --node-config 'partition-id=2 dataset-path="/custom/path/Dataset002_Heart" fold=2'
```

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

⚠️ **Important**: Models are only saved when the following conditions are met:
1. `OUTPUT_ROOT` environment variable is set
2. Validation is enabled (models save when validation improves)
3. Client achieves better validation performance than previous rounds

### Verifying Model Saving

After training completes, verify models were saved:
```bash
# Check if models were created
ls -la $OUTPUT_ROOT/client_*/model_best.pt

# Check model file sizes (should be >100MB for medical models)
du -sh $OUTPUT_ROOT/client_*/model_best.pt
```

## Example Commands

### Basic Federated Learning

```bash
# Simple 2-client setup
python run_federated_deployment.py --mode run --clients 2 --rounds 3
```

### Advanced Modality-Aware Setup

```bash
# CT and MR clients with custom weights
python run_federated_deployment.py --mode run \
    --dataset Dataset005_Prostate \
    --clients 4 --rounds 10 --local-epochs 3 \
    --enable-modality-aggregation \
    --modality-weights '{"CT": 0.6, "MR": 0.4}' \
    --validate --validation-frequency 2 \
    --output-dir "experiments/modality_aware"
```

### Distributed Setup (Multiple Machines)

```bash
# Machine 1 (Server): Start SuperLink
python run_federated_deployment.py --mode superlink \
    --superlink-host 0.0.0.0 --superlink-port 9091

# Machine 2 (Client 1): Start SuperNode
python run_federated_deployment.py --mode supernode \
    --superlink-host 192.168.1.100 --superlink-port 9091 \
    --node-id 0 --partition-id 0

# Machine 3 (Client 2): Start SuperNode  
python run_federated_deployment.py --mode supernode \
    --superlink-host 192.168.1.100 --superlink-port 9091 \
    --node-id 1 --partition-id 1

# Machine 1 (Server): Run Federation
flwr run . deployment
```

## Troubleshooting

### Common Issues

1. **SuperLink Connection Failed**
   - Check if SuperLink is running: `netstat -an | grep 9091`
   - Verify host/port configuration
   - Ensure firewall allows connections

2. **Client Connection Issues**
   - Wait 2-3 seconds between starting SuperLink and SuperNodes
   - Check SuperLink logs for client registration
   - Verify partition IDs are unique

3. **Modality Detection Issues**
   - Check `dataset.json` channel_names format
   - Verify `dataset_fingerprint.json` exists
   - Enable debug logging to see extracted modality info

4. **Memory Issues**
   - Reduce batch size or local epochs
   - Monitor GPU memory usage
   - Use gradient checkpointing if available

5. **Models Not Being Saved**
   - **Missing OUTPUT_ROOT**: Ensure `export OUTPUT_ROOT="/path/to/federated_models"`
   - **Validation Disabled**: Models only save during validation rounds with improvement
   - **Permissions**: Check write permissions: `mkdir -p $OUTPUT_ROOT && touch $OUTPUT_ROOT/test.txt`
   - **No Validation Improvement**: Models only save when validation Dice score improves
   - **Check Logs**: Look for "Saved best model checkpoint" in client logs

### Validation

- Check server logs for modality group formation
- Verify client modality assignments in preprocessing phase
- Monitor aggregation strategy selection (traditional vs modality-aware)

## Performance Optimization

- Use SSD storage for dataset access
- Enable GPU acceleration with `--gpu 0`
- Adjust `--local-epochs` based on available compute
- Use `--no-validate` for faster training (validation disabled)

## Next Steps

1. Experiment with different modality weight configurations
2. Add more imaging modalities (PET, US, etc.)
3. Implement domain adaptation techniques
4. Scale to larger numbers of clients
