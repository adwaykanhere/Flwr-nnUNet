# Federated nnU-Net with Flower Framework

This project implements a federated learning version of nnU-Net using the Flower framework. It enables distributed training of medical image segmentation models across multiple clients while keeping data decentralized and private.

## Overview

The implementation follows a 3-phase federated learning approach:
- **Phase -2**: Fingerprint collection from all clients
- **Phase -1**: Global initialization and parameter distribution  
- **Phase 0+**: Federated training rounds with model aggregation

## Key Features

- ✅ **Native nnUNet Integration**: Uses nnUNet's proven dataloaders, transforms, and training methods
- ✅ **Real Medical Data Support**: Handles nnU-Net v2 preprocessed data in standard .npz/.pkl format
- ✅ **Multi-Phase Federation**: Implements fingerprint collection, initialization, and training phases
- ✅ **GPU & CPU Support**: Optimized for both GPU acceleration and CPU-only environments
- ✅ **Deep Supervision**: Fully supports nnUNet's 6-level deep supervision architecture
- ✅ **Cross-Validation Support**: Maintains nnU-Net's 5-fold cross-validation splits
- ✅ **Any nnUNet Dataset**: Works with any nnUNet-compatible medical imaging dataset
- ✅ **Real Training Execution**: Performs actual training with loss computation and parameter updates
- ✅ **Real Properties Integration**: Uses actual medical imaging metadata and preserves data privacy

## Architecture

### Components

1. **`server_app.py`**: Implements `NnUNetFederatedStrategy` for coordinating the federated learning process
2. **`client_app.py`**: Handles client-side operations including fingerprint collection and local training
3. **`task.py`**: Custom `FedNnUNetTrainer` that extends nnU-Net's trainer for federated scenarios
4. **`pyproject.toml`**: Flower app configuration and federation settings

### Key Modifications

#### Native nnUNet Integration (`task.py`)
- **Native Dataloaders**: Uses nnUNet's `nnUNetDataLoader3D` and `nnUNetDataset` for proper data handling
- **Native Training Pipeline**: Leverages nnUNet's `train_step` method and epoch management
- **Native Transforms**: Uses nnUNet's proven data augmentation and preprocessing transforms
- **Deep Supervision**: Properly handles nnUNet's multi-scale segmentation outputs
- **Ray Compatibility**: Maintains compatibility with Ray distributed execution using single-threaded augmentation

#### Federated Strategy (`server_app.py`)
- **Fingerprint Aggregation**: Merges dataset fingerprints from multiple clients using weighted averaging
- **Parameter Distribution**: Handles global model initialization and updates
- **Round Management**: Coordinates the multi-phase training process

#### Client Implementation (`client_app.py`)
- **Phase-Aware Training**: Different behaviors for fingerprint, initialization, and training phases
- **Local Model Management**: Handles model weights serialization/deserialization
- **Metadata Exchange**: Shares dataset characteristics while preserving privacy

## Setup Instructions

### Prerequisites

1. **Python Environment**: Python 3.8+ with conda/pip
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
   pip install -r requirements.txt
   # or using the project dependencies:
   pip install "flwr[simulation]>=1.15.2" "flwr-datasets[vision]>=0.5.0"
   pip install torch==2.5.1 torchvision==0.20.1 numpy==1.26.4
   ```

3. **Install nnU-Net v2**
   ```bash
   # If not already installed
   pip install nnunetv2
   # OR install from source in development mode
   cd nnUNet && pip install -e .
   ```

4. **Setup Data Paths**
   
   Update the paths in your environment or modify the code to point to your preprocessed data:
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
   Your preprocessed data should have this structure:
   ```
   nnUNet_preprocessed/DatasetXXX_Name/
   ├── dataset.json
   ├── dataset_fingerprint.json
   ├── nnUNetPlans.json
   ├── splits_final.json
   └── nnUNetPlans_3d_fullres/
       ├── case_001.npz
       ├── case_001.pkl
       ├── case_002.npz
       ├── case_002.pkl
       └── ...
   ```

3. **Update Dataset Configuration**
   
   Modify `client_app.py` or set environment variables to point to your dataset:
   ```python
   # In client_app.py, update the task_name default:
   task_name = os.environ.get("TASK_NAME", "DatasetXXX_YourDataset")
   
   # Or set environment variable:
   export TASK_NAME="Dataset009_Spleen"  # Change to your dataset
   export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
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
   ```

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

1. **Start the Simulation**
   ```bash
   flwr run .
   ```

2. **Monitor Progress**
   The simulation will output logs showing:
   - Dataset loading and case discovery
   - Fingerprint collection from clients
   - Training round progress
   - Model aggregation results

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
   - Verify .npz files exist and are readable
   - Check that .pkl property files contain required fields like `class_locations`
   - Ensure dataset is properly preprocessed with nnUNetv2_plan_and_preprocess

### Performance Optimization

1. **GPU Acceleration**: For best performance, use GPU training:
   ```python
   device = torch.device("cuda")  
   ```

2. **CPU Optimization**: When using CPU, the system optimizes threading:
   ```python
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'
   os.environ['NUMEXPR_NUM_THREADS'] = '1'
   ```

3. **Native nnUNet Pipeline**: Uses nnUNet's optimized training methods for maximum efficiency

4. **Memory Usage**: Properties are cached to reduce I/O overhead during training

5. **Simulation Speed**: Reduce `num-server-rounds` for faster testing

## Recent Updates

### v3.0 - Native nnUNet Integration 
- ✅ **Native nnUNet Pipeline**: Completely replaced custom data loading with nnUNet's native dataloaders and transforms
- ✅ **Fixed Transform Pipeline**: Resolved `TypeError: argument after ** must be a mapping, not NoneType` by using nnUNet's native data format
- ✅ **Real Training Execution**: Now performs actual nnUNet training with real loss computation and parameter updates
- ✅ **Deep Supervision Support**: Properly handles nnUNet's 6-level deep supervision architecture
- ✅ **GPU Support**: Added comprehensive GPU support with proper CUDA configuration
- ✅ **Performance Optimization**: Leverages nnUNet's proven training methods for optimal performance

### v2.0 - Real Data Integration 
- ✅ **Fixed Pickle Loading Errors**: Resolved multiprocessing issues with dataset classes
- ✅ **Real Data Support**: Now loads actual nnUNet preprocessed .npz/.pkl files instead of dummy data
- ✅ **Generic Dataset Support**: Updated codebase to work with any nnUNet dataset, not just prostate
- ✅ **Improved Error Handling**: Better error messages and graceful handling of missing files
- ✅ **Updated API Compatibility**: Fixed Flower client API compatibility issues

### What's New in v3.0
The system now uses nnUNet's native training pipeline:
- **Before**: Custom data loading and training logic that sometimes failed
- **After**: Native nnUNet dataloaders, transforms, and training methods
- **Training Verification**: Real loss values printed (`Batch 1 loss: 1.1357`) confirming actual training
- **Data Shapes**: Proper medical imaging shapes `torch.Size([2, 2, 20, 320, 256])` with deep supervision
- **GPU Ready**: Full GPU support for faster training with proper CUDA handling

## Technical Details

### Native nnUNet Integration
- **Dataloaders**: Uses `nnUNetDataLoader3D` for proper 3D medical image handling
- **Transforms**: Native nnUNet augmentation pipeline with spatial transforms, intensity transforms, and deep supervision
- **Training**: Leverages nnUNet's `train_step` method with proper gradient scaling and loss computation
- **Architecture**: Supports nnUNet's U-Net with deep supervision (6 output scales)

### File Format Compatibility
- **Input**: nnU-Net v2 .npz preprocessed files (standard numpy format)
- **Properties**: .pkl files containing medical imaging metadata
- **Plans**: nnUNetPlans.json with 3d_fullres configuration
- **Deep Supervision**: Multi-scale targets at different resolutions

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
2. **Error Handling**: Add robust error handling for different data formats
3. **Testing**: Test with multiple datasets and cross-validation folds
4. **Documentation**: Update this README with any new features or requirements

## Acknowledgments

This implementation is based on:
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) for medical image segmentation
- [Flower Framework](https://flower.dev/) for federated learning
- [Kaapana](https://kaapana.readthedocs.io/en/latest/intro_kaapana.html) based Federated learning concepts from medical AI research

## License

This project follows the licensing terms of its dependencies:
- nnU-Net: Apache License 2.0
- Flower: Apache License 2.0
