# Federated nnU-Net with Flower Framework

This project implements a federated learning version of nnU-Net completely using the Flower framework. It enables distributed training of medical image segmentation models in a modality agnostic setting across multiple clients while keeping data decentralized and private.

## Architecture

### Components

1. **`run_federated.py`**: Enhanced standalone script with comprehensive CLI for federated training
2. **`run_federated_deployment.sh`**: Native bash script for SuperNode/SuperLink deployment with modality-aware aggregation support
3. **`server_app.py`**: Implements `NnUNetFederatedStrategy` for coordinating the federated learning process
4. **`server_app_modality.py`**: Enhanced server with `ModalityAwareFederatedStrategy` for multi-modal aggregation
5. **`client_app.py`**: Handles client-side operations including fingerprint collection, local training, modality detection, and model saving
6. **`task.py`**: Custom `FedNnUNetTrainer` that extends nnU-Net's trainer for federated scenarios with validation and PyTorch model saving
7. **`pyproject.toml`**: Flower app configuration and federation settings including deployment configurations

### Key Features

#### Native nnUNet Integration (`task.py`)
- **Unified DataLoader**: Uses nnUNet's `nnUNetDataLoader` with automatic 2D/3D detection
- **Automatic Dataset Detection**: Uses `infer_dataset_class()` to automatically detect B2ND or NPZ formats
- **B2ND & NPZ Support**: Seamlessly handles both compressed B2ND and standard NPZ data formats
- **Native Training Pipeline**: Leverages nnUNet's `train_step` method and epoch management
- **Native Transforms**: Uses nnUNet's proven data augmentation and preprocessing transforms
- **Deep Supervision**: Properly handles nnUNet's multi-scale segmentation outputs
- **Ray Compatibility**: Maintains compatibility with Ray distributed execution using single-threaded augmentation

#### Depployment-Ready Federated Strategy (`server_app.py` & `server_app_modality.py`)
- **Fingerprint Aggregation**: Merges dataset fingerprints from multiple clients using weighted averaging
- **Parameter Distribution**: Handles global model initialization and updates
- **Round Management**: Coordinates the multi-phase training process
- **Modality-Aware Aggregation**: Groups clients by detected modality (CT, MR, PET, US) for improved aggregation
- **Intra-Modal Aggregation**: First aggregates within modality groups
- **Inter-Modal Aggregation**: Weighted combination of modality-specific models into global model

#### Client-Side Modality-Agnostic Implementation (`client_app.py`)
- **Phase-Aware Training**: Different behaviors for fingerprint, initialization, and training phases
- **Local Model Management**: Handles model weights serialization/deserialization
- **Metadata Exchange**: Shares dataset characteristics while preserving privacy
- **Modality Detection**: Automatic extraction of imaging modality from dataset.json channel names
- **Enhanced Metadata**: Transmits modality information and dataset characteristics to server

## Setup Instructions

### Prerequisites

1. **Python Environment**: Python 3.10+ with conda/pip
2. **nnU-Net Installation**: nnU-Net v2 must be installed and configured
3. **Preprocessed Data**: Any nnU-Net preprocessed dataset in standard .npz/.pkl format
4. **Flower Framework**: Latest Flower with simulation support

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/adwaykanhere/Flwr-nnUNet.git
   cd Flwr-nnUNet
   ```

2. **Install Dependencies**
   ```bash
   conda create -n flwrtest python=3.10
   conda activate flwrtest
   pip install -r requirements.txt
   ```

3. **Create a copy of nnU-Net v2**
   ```bash
   # copy of the source repo of nnUNetv2
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   ```

4. **Setup Data Paths**
   
   Set the nnUNet preprocessed data environment variable:
   ```bash
   export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
   export nnUNet_raw="/path/to/your/nnUNet_raw"
   export nnUNet_results="/path/to/your/nnUNet_results"
   ```

### Data Preparation

1. **Preprocess Dataset with nnU-Net**
   ```bash
   nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
   ```

2. **Verify Data Structure**
   Your preprocessed data should have this structure (supports both B2ND and NPZ formats):
   ```
   nnUNet_preprocessed/DatasetXXX_Name/
   ├── dataset.json
   ├── dataset_fingerprint.json
   ├── nnUNetPlans.json
   ├── splits_final.json
   └── nnUNetPlans_3d_fullres/
       ├── case_001.b2nd          # Compressed format (preferred)
       ├── case_001_seg.b2nd
       ├── case_001.pkl           # Properties file
       ├── case_002.b2nd
       ├── case_002_seg.b2nd
       ├── case_002.pkl
       └── ...
       
   # OR alternatively with NPZ format:
   └── nnUNetPlans_3d_fullres/
       ├── case_001.npz           # Standard format
       ├── case_001.pkl           # Properties file
       ├── case_002.npz
       ├── case_002.pkl
       └── ...
   ```

3. **Configure Paths and Dataset**
   
   **Option A: Environment Variables (Recommended)**
   ```bash
   # Set your nnUNet preprocessed path
   export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
   
   # Set your dataset name
   export TASK_NAME="Dataset009_Spleen"  # Change to your dataset
   ```
   
   **Option B: Interactive Configuration**
   - The system will prompt you for paths if environment variables are not set
   - `test_trainer.py` will ask for the nnUNet_preprocessed path interactively
   - `client_app.py` will show warnings and use fallback paths with guidance
   
   **Option C: Direct Modification**
   ```python
   # In client_app.py, update the fallback path in get_nnunet_preprocessed_path()
   fallback_path = "/your/custom/nnUNet_preprocessed/"
   ```

### Configuration

1. **Federation Settings** (`pyproject.toml`)
   ```toml
   [tool.flwr.app.config]
   num-server-rounds = 2        # Number of training rounds
   fraction-fit = 1.0           # Fraction of clients for training
   fraction-evaluate = 0.0      # Fraction of clients for evaluation
   
   [tool.flwr.federations.local-simulation]
   options.num-supernodes = 2   # Number of simulated clients
   
   [tool.flwr.federations.supernode-deployment]
   address = "127.0.0.1:9093"                # SuperLink server address
   insecure = true                           # Use insecure connection for testing
   options.num-supernodes = 2                # Number of SuperNode clients
   options.enable-modality-aggregation = true # Enable modality-aware aggregation
   ```

📖 **For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

2. **Device Configuration** (`task.py`)
   ```python
   # For CPU-only training (default, Ray-compatible):
   device = torch.device("cpu")
   
   # For GPU training (faster, requires CUDA setup):
   device = torch.device("cuda")  # or torch.device("cuda:0")
   
   # Other key parameters:
   max_num_epochs = 50          # Max epochs per client
   fold = 0                     # Cross-validation fold (0-4)
   local_epochs_per_round = 2   # Epochs per federated round
   ```

### Running the Federated Training

You can run federated training in two ways:

#### Option 1: Flower Deployment (Recommended)

The native terminal approach using Flower's deployment commands directly:

1. **Activate Conda Environment**
   ```bash
   conda activate flwrtest  # or your preferred environment
   ```

2. **Start SuperLink (Terminal 1)**
   ```bash
   flower-superlink --insecure
   ```

3. **Start SuperNodes (Terminal 2-3)**
   ```bash
   # Terminal 2: First SuperNode (Client 0)
   flower-supernode --insecure --superlink 127.0.0.1:9092 \
       --clientappio-api-address 127.0.0.1:9094 \
       --node-config "partition-id=0"
   
   # Terminal 3: Second SuperNode (Client 1)
   flower-supernode --insecure --superlink 127.0.0.1:9092 \
       --clientappio-api-address 127.0.0.1:9095 \
       --node-config "partition-id=1"
   ```

4. **Run Federation (Terminal 4)**
   ```bash
   flwr run . deployment
   ```

#### Option 2: Automated Deployment Script (Alternative)

For convenience, use the bash script for automated deployment:

1. **Activate Conda Environment**
   ```bash
   conda activate flwrtest  # or your preferred environment
   ```

2. **Single Dataset Training**
   ```bash
   ./run_federated_deployment.sh --dataset Dataset005_Prostate --clients 2 --rounds 3 --local-epochs 2 --validate
   ```

3. **Multi-Dataset Training (with Modality-Aware Aggregation)**
   ```bash
   ./run_federated_deployment.sh --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}' --clients 2 --rounds 3 --enable-modality-aggregation
   ```

4. **Available Arguments**
   ```bash
   # Dataset selection
   --dataset Dataset005_Prostate              # Single dataset for all clients
   --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}' # Multi-dataset mapping
   --list-datasets                            # List available datasets
   --validate-datasets                        # Validate multi-dataset compatibility
   
   # Training configuration  
   --clients 2                                # Number of federated clients
   --rounds 5                                 # Number of federated learning rounds
   --local-epochs 2                           # Local epochs per client per round
   
   # Modality-aware aggregation
   --enable-modality-aggregation              # Enable modality-aware aggregation
   --modality-weights '{"CT": 0.6, "MR": 0.4}' # Custom modality weights
   
   # Deployment configuration
   --mode run                                 # Mode: superlink, supernode, or run (full deployment)
   --superlink-host 127.0.0.1                # SuperLink host address
   --superlink-port 9091                      # SuperLink port
   --insecure                                 # Use insecure connection (for testing)
   
   # Validation options
   --validate                                 # Enable validation during training (default)
   --no-validate                              # Skip validation for faster training
   --validation-frequency 1                   # Validate every N rounds
   
   # Model saving
   --output-dir federated_models              # Output directory for saved models
   --save-frequency 1                         # Save models every N rounds
   
   # System configuration
   --gpu 0                                    # GPU device ID to use
   ```

5. **Example Commands**
   ```bash
   # List available datasets
   ./run_federated_deployment.sh --list-datasets
   
   # Validate multi-dataset compatibility
   ./run_federated_deployment.sh --validate-datasets --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}'
   
   # Quick test run with minimal epochs
   ./run_federated_deployment.sh --dataset Dataset005_Prostate --clients 2 --rounds 1 --local-epochs 1
   
   # Multi-dataset training with modality-aware aggregation
   ./run_federated_deployment.sh --client-datasets '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset002_Heart"}' --clients 3 --rounds 5 --enable-modality-aggregation
   
   # Training without validation for speed
   ./run_federated_deployment.sh --dataset Dataset005_Prostate --clients 2 --rounds 5 --no-validate
   ```

### Multi-Dataset Federation

📖 **For detailed multi-dataset instructions, see [MULTI_DATASET_GUIDE.md](MULTI_DATASET_GUIDE.md)**

### Modality-Aware Aggregation

The system automatically detects client modalities from dataset.json:
- **CT**: Channel names containing "ct", "computed"
- **MR**: Channel names containing "mr", "magnetic", "t1", "t2"  
- **PET**: Channel names containing "pet"
- **US**: Channel names containing "us", "ultrasound"

**Aggregation Strategy**:
- **Intra-modality**: CT clients aggregate → CT model, MR clients aggregate → MR model
- **Inter-modality**: Weighted combination of modality models → Global model
- **Fallback**: Traditional FedAvg when modality aggregation is disabled

## GPU Usage

### Enabling GPU Training

The system supports both CPU and GPU training. For significantly faster performance, use GPU:

#### 1. **Prerequisites for GPU Training**
```bash
# Ensure CUDA is properly installed
nvidia-smi  # Should show your GPU(s)

# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 2. **Configure GPU Usage**

**Option A: Modify the code directly in `task.py`:**
```python
# Change this line in FedNnUNetTrainer.__init__:
device: torch.device = torch.device("cuda")  # Instead of "cpu"

# Also remove the CUDA disable line:
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Comment out or remove this line
```

**Option B: Set environment variables:**
```bash
# Enable specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# Or enable multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

#### 3. **GPU Memory Management**

For large medical images, you may need to adjust batch sizes:
```python
# In your configuration, reduce batch size if you get CUDA OOM errors
batch_size = 1  # Instead of default 2
```

#### 4. **Mixed Precision Training**

nnUNet supports automatic mixed precision for faster GPU training:
```python
# This is handled automatically by nnUNet's native training pipeline
# Mixed precision is enabled based on device capabilities
```

### GPU Troubleshooting

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in configuration manager
   self.batch_size = 1  # Instead of 2
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version compatibility
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Multiple GPU Issues**
   ```bash
   # Use specific GPU
   export CUDA_VISIBLE_DEVICES=0
   ```

4. **Mixed Environment Issues**
   ```python
   # Force CPU if needed
   os.environ['CUDA_VISIBLE_DEVICES'] = ''
   ```

## Troubleshooting

### Common Issues

1. **CUDA Crashes in WSL2**
   - The code disables CUDA by default via environment variables
   - If you encounter crashes, ensure no other processes are using CUDA

2. **Memory Issues with Large Datasets**
   - The system preloads properties to minimize file I/O
   - For very large datasets, consider reducing `num-supernodes`

3. **Import Errors**
   - Ensure nnU-Net is properly installed: `pip show nnunetv2`
   - Check that all paths are correctly set in environment variables

4. **Data Loading Failures**
   - **B2ND Format**: Ensure `blosc2` is installed: `pip install blosc2`
   - **NPZ Format**: Verify .npz files exist and are readable
   - **Mixed Formats**: Don't mix B2ND and NPZ files in the same dataset folder
   - **Properties**: Check that .pkl property files contain required fields like `class_locations`
   - **Preprocessing**: Ensure dataset is properly preprocessed with nnUNetv2_plan_and_preprocess

5. **Path Configuration Issues**
   - **Environment Variable**: Ensure `nnUNet_preprocessed` is set: `echo $nnUNet_preprocessed`
   - **Interactive Mode**: When prompted, provide the full absolute path
   - **Permissions**: Ensure read access to the nnUNet_preprocessed directory
   - **Dataset Name**: Verify `TASK_NAME` environment variable matches your dataset folder

6. **Models Not Being Saved**
   - **Missing OUTPUT_ROOT**: Set `export OUTPUT_ROOT="/path/to/federated_models"`
   - **Validation Disabled**: Models are only saved during validation - ensure validation is enabled
   - **Directory Permissions**: Check write permissions for the output directory
   - **Verification**: After training, check for `.pt` files in `OUTPUT_ROOT/client_X/` directories
   - **Default Path**: If not set, defaults to `/local/projects-t3/isaiahlab/nnunet_output`

7. **B2ND Format Issues**
   - **Missing blosc2**: Install with `pip install blosc2`
   - **Corruption**: B2ND files may be corrupted, rerun preprocessing
   - **Version Mismatch**: Ensure nnUNet v2 latest version for B2ND support

## Technical Details

### Native nnUNet Integration
- **Dataloaders**: Uses `nnUNetDataLoader3D` for proper 3D medical image handling
- **Transforms**: Native nnUNet augmentation pipeline with spatial transforms, intensity transforms, and deep supervision
- **Training**: Leverages nnUNet's `train_step` method with proper gradient scaling and loss computation
- **Architecture**: Supports nnUNet's U-Net with deep supervision (6 output scales)

### File Format Compatibility
- **Input Data**: Follows nnUNet data formats
  - `.b2nd` files (compressed format, preferred for performance)
  - `.npz` files (standard numpy format, legacy support)
  - Automatic format detection and loading
- **Properties**: `.pkl` files containing medical imaging metadata
- **Plans**: `nnUNetPlans.json` with 3d_fullres configuration
- **Deep Supervision**: Multi-scale targets at different resolutions
- **Format Detection**: Uses `infer_dataset_class()` for automatic B2ND/NPZ detection

### Federated Learning Process
1. **Fingerprint Phase**: Clients share dataset statistics (shapes, spacings, intensity properties)
2. **Initialization Phase**: Server distributes initial model parameters
3. **Training Phases**: Iterative local training with native nnUNet methods and global aggregation

### Performance Features
- **GPU Support**: Full CUDA acceleration with automatic mixed precision
- **Ray Compatibility**: Single-threaded augmentation for distributed execution
- **Memory Optimization**: Efficient data loading and caching strategies
- **Native Training**: Uses nnUNet's proven training pipeline for optimal convergence

### Security Considerations
- Only aggregated statistics are shared between clients
- Raw medical imaging data never leaves the local environment
- Model parameters are the only data transmitted

## Contributing

When extending this implementation:

1. **Maintain Privacy**: Ensure no raw data is transmitted between clients
2. **Error Handling**: Add robust handling for different data formats
3. **Testing**: Test with multiple datasets and cross-validation folds
4. **Documentation**: Update this README with any new features or requirements

## Acknowledgments

This implementation is based on:
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) for medical image segmentation
- [Flower Framework](https://flower.dev/) for setting up the complete federated learning infrastructure
- [Kaapana](https://kaapana.readthedocs.io/en/latest/intro_kaapana.html) based Federated learning concepts were used while federating nnUNet 

## License

This project follows the licensing terms of its dependencies:
- nnU-Net: Apache License 2.0
- Flower: Apache License 2.0
