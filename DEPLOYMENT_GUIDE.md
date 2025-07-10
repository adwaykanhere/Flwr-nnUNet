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

### Option 1: Automated Full Deployment

Run the complete federated learning setup automatically:

```bash
# Activate conda environment
conda activate flwrtest

# Basic deployment with 2 clients
python run_federated_deployment.py --dataset Dataset005_Prostate --clients 2 --rounds 3

# With modality-aware aggregation
python run_federated_deployment.py --dataset Dataset005_Prostate --clients 2 --rounds 3 \
    --enable-modality-aggregation \
    --modality-weights '{"CT": 0.6, "MR": 0.4}'

# Multi-dataset federation (different datasets per client)
python run_federated_deployment.py \
    --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}' \
    --clients 3 --rounds 5 --local-epochs 2 \
    --enable-modality-aggregation \
    --validate
```

### Option 2: Manual Step-by-Step Deployment

#### Step 1: Start SuperLink (Server)

```bash
# Terminal 1: Start the SuperLink server
flower-superlink --insecure
```

#### Step 2: Start SuperNodes (Clients)

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

#### Step 3: Run Federation

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
| `--partition-id` | `0` | Client partition ID |
| `--insecure` | `True` | Use insecure connection (for testing) |

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
export NUM_CLIENTS=2
export NUM_TRAINING_ROUNDS=3
export LOCAL_EPOCHS=2
export ENABLE_MODALITY_AGGREGATION=true
export MODALITY_WEIGHTS='{"CT": 0.6, "MR": 0.4}'
export OUTPUT_DIR="federated_models"
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

Models are saved to the specified output directory:

```
federated_models/
├── Dataset005_Prostate/
│   ├── server/
│   │   ├── global_best_model_modality_aware.pt
│   │   └── round_X_metadata.json
│   ├── client_0/
│   │   └── model_best.pt
│   └── client_1/
│       └── model_best.pt
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