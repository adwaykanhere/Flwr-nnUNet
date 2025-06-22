# Federated nnU-Net with Flower Framework

This project implements a federated learning version of nnU-Net using the Flower framework, similar to the approach used in the Kaapana platform. It enables distributed training of medical image segmentation models while keeping data decentralized.

## Overview

The implementation follows a 3-phase federated learning approach:
- **Phase -2**: Fingerprint collection from all clients
- **Phase -1**: Global initialization and parameter distribution  
- **Phase 0+**: Federated training rounds with model aggregation

## Key Features

- ✅ **Real Medical Data Support**: Handles nnU-Net v2 preprocessed data in .b2nd compressed format
- ✅ **Kaapana-Style Federation**: Implements the multi-phase approach used in clinical federated learning
- ✅ **CPU-Only Execution**: Optimized for environments without GPU access or CUDA issues
- ✅ **Cross-Validation Support**: Maintains nnU-Net's 5-fold cross-validation splits
- ✅ **Real Properties Integration**: Uses actual medical imaging metadata (spacing, class_locations, etc.)

## Architecture

### Components

1. **`server_app.py`**: Implements `KaapanaStyleStrategy` for coordinating the federated learning process
2. **`client_app.py`**: Handles client-side operations including fingerprint collection and local training
3. **`task.py`**: Custom `FedNnUNetTrainer` that extends nnU-Net's trainer for federated scenarios
4. **`pyproject.toml`**: Flower app configuration and federation settings

### Key Modifications

#### Custom Data Loading (`task.py`)
- **B2ND File Support**: Custom dataset loader for nnU-Net's compressed .b2nd files using blosc2
- **Property Caching**: Preloads all .pkl properties files to avoid I/O issues during training
- **Safe Error Handling**: Graceful fallbacks when individual files fail to load
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
3. **Preprocessed Data**: nnU-Net preprocessed dataset in .b2nd format
4. **Flower Framework**: Latest Flower with simulation support

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
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
       ├── case_001.b2nd
       ├── case_001.pkl
       ├── case_001_seg.b2nd
       ├── case_002.b2nd
       ├── case_002.pkl
       ├── case_002_seg.b2nd
       └── ...
   ```

3. **Update Dataset Configuration**
   
   Modify `client_app.py` to point to your dataset:
   ```python
   # Update these paths in client_app.py
   dataset_base = "/path/to/nnUNet_preprocessed"
   dataset_name = "Dataset005_Prostate"  # Change to your dataset
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
   [Server] Starting kaapana-style federated learning
   [Trainer] Found 32 case identifiers: ['prostate_00', 'prostate_01', ...]
   [Trainer] Creating B2ND datasets with real prostate data - tr: 25, val: 7
   [Dataset] Preloading properties for 25 cases...
   ```

## Troubleshooting

### Common Issues

1. **CUDA Crashes in WSL2**
   - The code disables CUDA by default via environment variables
   - If you encounter crashes, ensure no other processes are using CUDA

2. **Memory Issues with blosc2**
   - The system preloads properties to minimize file I/O
   - For very large datasets, consider reducing `num-supernodes`

3. **Import Errors**
   - Ensure nnU-Net is properly installed: `pip show nnunetv2`
   - Check that all paths are correctly set in environment variables

4. **Data Loading Failures**
   - Verify .b2nd files exist and are readable
   - Check that .pkl property files contain required fields like `class_locations`

### Performance Optimization

1. **CPU-Only Mode**: The system is optimized for CPU execution with disabled threading:
   ```python
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'
   os.environ['NUMEXPR_NUM_THREADS'] = '1'
   ```

2. **Memory Usage**: Properties are cached to reduce I/O overhead during training

3. **Simulation Speed**: Reduce `num-server-rounds` for faster testing

## Technical Details

### File Format Compatibility
- **Input**: nnU-Net v2 .b2nd compressed files (blosc2 format)
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
- [Kaapana Platform](https://kaapana.ai/) for the federated learning strategy

## License

This project follows the licensing terms of its dependencies:
- nnU-Net: Apache License 2.0
- Flower: Apache License 2.0
