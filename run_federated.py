#!/usr/bin/env python3
"""
Enhanced Federated nnUNet Implementation
Supports dataset selection, model saving, and validation with Dice score calculation
"""

import os
# Set environment before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['nnUNet_n_proc_DA'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Flower based Federated nnUNet Training')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='Dataset005_Prostate',
                       help='Dataset name (e.g., Dataset005_Prostate, Dataset009_Spleen)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    
    # Training configuration
    parser.add_argument('--clients', type=int, default=2,
                       help='Number of federated clients')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of federated learning rounds')
    parser.add_argument('--local-epochs', type=int, default=1,
                       help='Local epochs per client per round')
    
    # Validation options
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Run validation during training')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation (faster training)')
    parser.add_argument('--validation-frequency', type=int, default=1,
                       help='Validate every N rounds (default: every round)')
    
    # Model saving options
    parser.add_argument('--output-dir', type=str, default='federated_models',
                       help='Output directory for saved models')
    parser.add_argument('--save-frequency', type=int, default=1,
                       help='Save models every N rounds (default: every round)')
    
    # System configuration
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    
    return parser.parse_args()

def list_available_datasets(preproc_root: str) -> List[str]:
    """List all available preprocessed datasets"""
    if not os.path.exists(preproc_root):
        return []
    
    datasets = []
    for item in os.listdir(preproc_root):
        item_path = os.path.join(preproc_root, item)
        if os.path.isdir(item_path) and item.startswith('Dataset'):
            # Check if it has required files
            plans_file = os.path.join(item_path, 'nnUNetPlans.json')
            dataset_file = os.path.join(item_path, 'dataset.json')
            if os.path.exists(plans_file) and os.path.exists(dataset_file):
                datasets.append(item)
    
    return sorted(datasets)

def validate_dataset(dataset_name: str, preproc_root: str) -> bool:
    """Validate that the selected dataset exists and has required files"""
    dataset_path = os.path.join(preproc_root, dataset_name)
    
    if not os.path.exists(dataset_path):
        return False
    
    required_files = ['nnUNetPlans.json', 'dataset.json']
    for file_name in required_files:
        if not os.path.exists(os.path.join(dataset_path, file_name)):
            return False
    
    return True

print("=== Flower based nnUNet Federated Learning ===")

# Parse arguments
args = parse_arguments()

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Define paths
preproc_root = '/Users/akanhere/Documents/nnUNet/nnUNet_preprocessed'

# Handle list datasets option
if args.list_datasets:
    print("\nAvailable datasets:")
    datasets = list_available_datasets(preproc_root)
    if datasets:
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")
    else:
        print("  No datasets found in", preproc_root)
    exit(0)

# Validate selected dataset
if not validate_dataset(args.dataset, preproc_root):
    available = list_available_datasets(preproc_root)
    print(f"❌ Dataset '{args.dataset}' not found or invalid.")
    print(f"Available datasets: {', '.join(available)}")
    exit(1)

print(f"📊 Dataset: {args.dataset}")
print(f"👥 Clients: {args.clients}")
print(f"🔄 Rounds: {args.rounds}")
print(f"📈 Local epochs: {args.local_epochs}")
print(f"✅ Validation: {'enabled' if args.validate and not args.no_validate else 'disabled'}")

# Override validation setting
validate_enabled = args.validate and not args.no_validate

try:
    from client_app import NnUNet3DFullresClient
    import torch
    
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"🎮 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💻 GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # Configuration from arguments
    NUM_CLIENTS = args.clients
    NUM_ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    
    # Set up paths using selected dataset
    task_name = args.dataset
    plans_path = os.path.join(preproc_root, task_name, 'nnUNetPlans.json')
    dataset_json = os.path.join(preproc_root, task_name, 'dataset.json')
    
    # Create output directory structure
    output_base = Path(args.output_dir)
    dataset_output = output_base / task_name
    server_output = dataset_output / "server"
    
    # Create directories if they don't exist
    server_output.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_CLIENTS):
        client_output = dataset_output / f"client_{i}"
        client_output.mkdir(parents=True, exist_ok=True)
    
    def save_model_with_metadata(model_params: List[np.ndarray], metadata: Dict[str, Any], 
                                output_dir: Path, round_num: int, model_type: str = "model"):
        """Save model parameters and metadata"""
        # Save model parameters as .npz
        model_file = output_dir / f"round_{round_num}_{model_type}.npz"
        param_dict = {f"param_{i}": param for i, param in enumerate(model_params)}
        np.savez_compressed(model_file, **param_dict)
        
        # Save metadata as JSON
        metadata_file = output_dir / f"round_{round_num}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"💾 Saved {model_type} to {model_file}")
        return model_file, metadata_file

    def load_saved_model(model_file: Path) -> List[np.ndarray]:
        """Load model parameters from saved file"""
        data = np.load(model_file)
        params = []
        i = 0
        while f"param_{i}" in data:
            params.append(data[f"param_{i}"])
            i += 1
        return params

    def create_client(client_id: int):
        """Create a federated client"""
        client = NnUNet3DFullresClient(
            client_id=client_id,
            plans_path=plans_path,
            dataset_json=dataset_json,
            configuration='3d_fullres',
            max_total_epochs=NUM_ROUNDS * LOCAL_EPOCHS,
            local_epochs_per_round=LOCAL_EPOCHS
        )
        # Set output directory for PyTorch model saving
        client.set_output_directory(str(output_base))
        return client
    
    def federated_averaging(client_params: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Simple FedAvg implementation"""
        if not client_params:
            return []
            
        # Get total number of examples
        total_examples = sum(num_examples for _, num_examples in client_params)
        
        # Initialize averaged parameters with zeros
        params_list = [params for params, _ in client_params]
        avg_params = []
        
        for i in range(len(params_list[0])):
            weighted_sum = np.zeros_like(params_list[0][i])
            for params, num_examples in client_params:
                weight = num_examples / total_examples
                weighted_sum += weight * params[i]
            avg_params.append(weighted_sum)
            
        return avg_params
    
    # Create clients
    print(f"\\nCreating {NUM_CLIENTS} clients...")
    clients = []
    for i in range(NUM_CLIENTS):
        print(f"Creating client {i}...")
        client = create_client(i)
        clients.append(client)
        print(f"✓ Client {i} created")
    
    # Get initial parameters from first client
    print("\\nInitializing global model...")
    global_params = clients[0].get_parameters({})
    print(f"✓ Global model initialized with {len(global_params)} parameters")
    
    # Federated learning rounds
    for round_num in range(NUM_ROUNDS):
        print(f"\\n=== Round {round_num + 1}/{NUM_ROUNDS} ===")
        
        client_results = []
        client_metrics = []
        
        # Train on each client
        for i, client in enumerate(clients):
            print(f"Training client {i}...")
            
            config = {
                'server_round': round_num,
                'local_epochs': LOCAL_EPOCHS,
                'validate': validate_enabled and (round_num + 1) % args.validation_frequency == 0
            }
            
            try:
                updated_params, num_examples, metrics = client.fit(global_params, config)
                client_results.append((updated_params, num_examples))
                client_metrics.append(metrics)
                
                loss = metrics.get('loss', 'N/A')
                val_dice = metrics.get('validation_dice', {})
                if val_dice:
                    dice_mean = val_dice.get('mean', 'N/A')
                    print(f"✓ Client {i}: {num_examples} examples, loss={loss}, dice={dice_mean}")
                else:
                    print(f"✓ Client {i}: {num_examples} examples, loss={loss}")
                
                # Save client model if requested
                if (round_num + 1) % args.save_frequency == 0:
                    client_output = dataset_output / f"client_{i}"
                    client_metadata = {
                        "client_id": i,
                        "round": round_num + 1,
                        "dataset": task_name,
                        "local_epochs": LOCAL_EPOCHS,
                        "training_examples": num_examples,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_model_with_metadata(updated_params, client_metadata, 
                                           client_output, round_num + 1, "local_model")
                
            except Exception as e:
                print(f"✗ Client {i} failed: {e}")
                continue
        
        # Aggregate results
        if client_results:
            print("Performing federated averaging...")
            global_params = federated_averaging(client_results)
            total_examples = sum(num_examples for _, num_examples in client_results)
            
            # Calculate aggregated metrics
            avg_loss = np.mean([m.get('loss', 0) for m in client_metrics])
            
            # Aggregate validation dice if available
            validation_dice = {}
            if any('validation_dice' in m for m in client_metrics):
                dice_scores = [m.get('validation_dice', {}) for m in client_metrics if 'validation_dice' in m]
                if dice_scores:
                    # Average dice scores across clients
                    all_means = [d.get('mean', 0) for d in dice_scores if 'mean' in d]
                    if all_means:
                        validation_dice['mean'] = np.mean(all_means)
                        
                        # Aggregate per-label dice scores
                        all_labels = set()
                        for d in dice_scores:
                            if 'per_label' in d:
                                all_labels.update(d['per_label'].keys())
                        
                        if all_labels:
                            validation_dice['per_label'] = {}
                            for label in all_labels:
                                label_scores = [d['per_label'].get(label, 0) for d in dice_scores 
                                              if 'per_label' in d and label in d['per_label']]
                                if label_scores:
                                    validation_dice['per_label'][label] = np.mean(label_scores)
            
            print(f"✓ Global model updated with {total_examples} total examples")
            print(f"📊 Average loss: {avg_loss:.4f}")
            if validation_dice and 'mean' in validation_dice:
                print(f"🎯 Average Dice: {validation_dice['mean']:.4f}")
            
            # Save global model if requested
            if (round_num + 1) % args.save_frequency == 0:
                server_metadata = {
                    "round": round_num + 1,
                    "dataset": task_name,
                    "total_clients": NUM_CLIENTS,
                    "successful_clients": len(client_results),
                    "total_examples": total_examples,
                    "average_loss": avg_loss,
                    "timestamp": datetime.now().isoformat()
                }
                
                if validation_dice:
                    server_metadata["validation_dice"] = validation_dice
                
                save_model_with_metadata(global_params, server_metadata, 
                                       server_output, round_num + 1, "global_model")
                
                # Also save PyTorch model for global model if validation was performed
                if validation_dice and 'mean' in validation_dice:
                    try:
                        # Use the first client's trainer to save the global model in PyTorch format
                        global_dice = validation_dice['mean']
                        pytorch_checkpoint = clients[0].trainer.save_best_checkpoint_pytorch(
                            output_dir=str(server_output),
                            round_num=round_num + 1,
                            validation_dice=global_dice,
                            is_best=False  # We'll track this separately
                        )
                        if pytorch_checkpoint:
                            print(f"💾 Saved global PyTorch checkpoint: {pytorch_checkpoint}")
                    except Exception as e:
                        print(f"⚠️ Failed to save global PyTorch checkpoint: {e}")
        else:
            print("❌ No clients completed training this round")
            break
    
    print(f"\\n🎉 Federated learning completed successfully!")
    print(f"Final model trained on data from {NUM_CLIENTS} clients over {NUM_ROUNDS} rounds")
    
except Exception as e:
    print(f"\\n❌ Error: {e}")
    import traceback
    traceback.print_exc()