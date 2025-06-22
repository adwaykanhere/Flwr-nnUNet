# Federated nnU-Net with Flower Framework

This project implements a federated learning version of nnU-Net using the Flower framework. It enables distributed training of medical image segmentation models across multiple clients while keeping data decentralized and private.

## Overview

The implementation follows a 3-phase federated learning approach:
- **Phase -2**: Fingerprint collection from all clients
- **Phase -1**: Global initialization and parameter distribution  
- **Phase 0+**: Federated training rounds with model aggregation

## Key Features

- ✅ **Real Medical Data Support**: Handles nnU-Net v2 preprocessed data in standard .npz/.pkl format
- ✅ **Multi-Phase Federation**: Implements fingerprint collection, initialization, and training phases
- ✅ **CPU-Only Execution**: Optimized for environments without GPU access or CUDA issues
- ✅ **Cross-Validation Support**: Maintains nnU-Net's 5-fold cross-validation splits
- ✅ **Any nnUNet Dataset**: Works with any nnUNet-compatible medical imaging dataset
- ✅ **Real Properties Integration**: Uses actual medical imaging metadata and preserves data privacy

## Architecture

### Components

1. **`server_app.py`**: Implements `NnUNetFederatedStrategy` for coordinating the federated learning process
2. **`client_app.py`**: Handles client-side operations including fingerprint collection and local training
3. **`task.py`**: Custom `FedNnUNetTrainer` that extends nnU-Net's trainer for federated scenarios
4. **`pyproject.toml`**: Flower app configuration and federation settings

### Key Modifications

#### Custom Data Loading (`task.py`)
- **nnUNet Format Support**: Custom dataset loader for nnU-Net's standard .npz/.pkl preprocessed files
- **Property Caching**: Preloads all .pkl properties files to avoid I/O issues during training
- **Real Data Integration**: Works with actual nnUNet preprocessed datasets without dummy data
- **CPU Optimization**: Aggressive multiprocessing disabling to prevent crashes

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

2. **Trainer Settings** (`task.py`)
   ```python
   # Key parameters you can modify:
   max_num_epochs = 50          # Max epochs per client
   device = torch.device("cpu") # Force CPU usage
   fold = 0                     # Cross-validation fold (0-4)
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

3. **Expected Output**
   ```
   [Server] Starting nnUNet federated learning
   [Trainer] Found 32 case identifiers: ['case_00', 'case_01', ...]
   [Trainer] Creating nnUNet datasets with real medical data - tr: 25, val: 7
   [Dataset] Preloading properties for 25 cases...
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

1. **CPU-Only Mode**: The system is optimized for CPU execution with disabled threading:
   ```python
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'
   os.environ['NUMEXPR_NUM_THREADS'] = '1'
   ```

2. **Memory Usage**: Properties are cached to reduce I/O overhead during training

3. **Simulation Speed**: Reduce `num-server-rounds` for faster testing

## Recent Updates

### v2.0 - Real Data Integration (June 2025)
- ✅ **Fixed Pickle Loading Errors**: Resolved multiprocessing issues with dataset classes
- ✅ **Real Data Support**: Now loads actual nnUNet preprocessed .npz/.pkl files instead of dummy data
- ✅ **Generic Dataset Support**: Updated codebase to work with any nnUNet dataset, not just prostate
- ✅ **Improved Error Handling**: Better error messages and graceful handling of missing files
- ✅ **Updated API Compatibility**: Fixed Flower client API compatibility issues

### Migration from Dummy Data
The system now processes real medical imaging data:
- **Before**: Used placeholder/dummy data for testing
- **After**: Loads actual nnUNet preprocessed files with real medical imaging properties
- **Datasets Tested**: Prostate (Dataset005), Spleen (Dataset009)

## Technical Details

### File Format Compatibility
- **Input**: nnU-Net v2 .npz preprocessed files (standard numpy format)
- **Properties**: .pkl files containing medical imaging metadata
- **Plans**: nnUNetPlans.json with 3d_fullres configuration

### Federated Learning Process
1. **Fingerprint Phase**: Clients share dataset statistics (shapes, spacings, intensity properties)
2. **Initialization Phase**: Server distributes initial model parameters
3. **Training Phases**: Iterative local training and global aggregation

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
