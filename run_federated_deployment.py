#!/usr/bin/env python3
"""
Enhanced Federated nnUNet Implementation using Flower SuperNode/SuperLink Deployment
Supports modality-aware federated averaging, dataset selection, model saving, and validation with Dice score calculation
"""

import os
import argparse
import json
import subprocess
import time
import signal
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Flower SuperNode/SuperLink based Federated nnUNet Training')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='Dataset005_Prostate',
                       help='Dataset name (e.g., Dataset005_Prostate, Dataset009_Spleen) - used when all clients use same dataset')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--client-datasets', type=str, default=None,
                       help='JSON mapping of client IDs to datasets (e.g., \'{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}\')')
    parser.add_argument('--validate-datasets', action='store_true', default=False,
                       help='Validate dataset compatibility for multi-dataset federation')
    
    # Training configuration
    parser.add_argument('--clients', type=int, default=2,
                       help='Number of federated clients')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of federated learning rounds')
    parser.add_argument('--local-epochs', type=int, default=1,
                       help='Local epochs per client per round')
    
    # Deployment configuration
    parser.add_argument('--mode', type=str, choices=['superlink', 'supernode', 'run'], default='run',
                       help='Deployment mode: superlink (server), supernode (client), or run (full deployment)')
    parser.add_argument('--superlink-host', type=str, default='127.0.0.1',
                       help='SuperLink host address')
    parser.add_argument('--superlink-port', type=int, default=9091,
                       help='SuperLink port')
    parser.add_argument('--node-id', type=int, default=0,
                       help='SuperNode ID (for client mode)')
    parser.add_argument('--partition-id', type=int, default=0,
                       help='Client partition ID')
    parser.add_argument('--insecure', action='store_true', default=True,
                       help='Use insecure connection (for testing)')
    
    # Modality-aware aggregation
    parser.add_argument('--enable-modality-aggregation', action='store_true', default=False,
                       help='Enable modality-aware federated averaging')
    parser.add_argument('--modality-weights', type=str, default=None,
                       help='JSON string of modality weights (e.g., \'{"CT": 0.6, "MR": 0.4}\')')
    
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

def parse_client_datasets(client_datasets_str: str) -> Dict[str, str]:
    """Parse client-dataset mapping from JSON string"""
    try:
        client_datasets = json.loads(client_datasets_str)
        # Ensure all keys are strings (client IDs)
        return {str(k): str(v) for k, v in client_datasets.items()}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing client-datasets JSON: {e}")
        return {}

def validate_multi_dataset_setup(client_datasets: Dict[str, str], preproc_root: str) -> bool:
    """Validate multi-dataset federation setup"""
    print("🔍 Validating multi-dataset federation setup...")
    
    all_valid = True
    unique_datasets = set(client_datasets.values())
    
    print(f"📊 Client-Dataset Mapping:")
    for client_id, dataset_name in client_datasets.items():
        print(f"  Client {client_id}: {dataset_name}")
    
    print(f"📈 Unique datasets: {len(unique_datasets)}")
    print(f"🏥 Total clients: {len(client_datasets)}")
    
    # Validate each dataset
    for client_id, dataset_name in client_datasets.items():
        if not validate_dataset(dataset_name, preproc_root):
            print(f"❌ Dataset '{dataset_name}' for client {client_id} is invalid or missing")
            all_valid = False
        else:
            print(f"✅ Dataset '{dataset_name}' for client {client_id} is valid")
    
    if len(unique_datasets) > 1:
        print("🔄 Multi-dataset federation detected")
        try:
            from dataset_compatibility import create_multi_dataset_config
            config = create_multi_dataset_config(client_datasets, preproc_root)
            
            print("\n📋 Compatibility Analysis:")
            compatibility = config.get('compatibility_analysis', {})
            modality_analysis = compatibility.get('modality_analysis', {})
            
            if modality_analysis:
                print(f"🧠 Detected modalities: {modality_analysis.get('unique_modalities', [])}")
                print(f"🏷️  Modality distribution: {modality_analysis.get('modality_counts', {})}")
            
            recommendations = compatibility.get('recommendations', [])
            if recommendations:
                print("\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"  {rec}")
            
            validation_report = config.get('validation_report', {})
            if not validation_report.get('valid', True):
                print("\n⚠️  Validation Issues:")
                for error in validation_report.get('errors', []):
                    print(f"  ❌ {error}")
                for warning in validation_report.get('warnings', []):
                    print(f"  ⚠️  {warning}")
                all_valid = False
            
        except ImportError:
            print("⚠️  Dataset compatibility analysis not available (dataset_compatibility.py not found)")
        except Exception as e:
            print(f"⚠️  Error during compatibility analysis: {e}")
    
    return all_valid

def setup_multi_dataset_environment(client_datasets: Dict[str, str], args):
    """Set up environment variables for multi-dataset federation"""
    # Set the client-datasets mapping
    os.environ['CLIENT_DATASETS'] = json.dumps(client_datasets)
    
    # If only one unique dataset, use it as the global dataset
    unique_datasets = set(client_datasets.values())
    if len(unique_datasets) == 1:
        os.environ['TASK_NAME'] = list(unique_datasets)[0]
        print(f"📊 Single dataset federation: {list(unique_datasets)[0]}")
    else:
        # Multi-dataset federation
        print(f"🔄 Multi-dataset federation: {len(unique_datasets)} datasets across {len(client_datasets)} clients")
        
        # Automatically enable modality aggregation for multi-dataset setups
        if not args.enable_modality_aggregation:
            print("🧠 Automatically enabling modality-aware aggregation for multi-dataset federation")
            args.enable_modality_aggregation = True
            os.environ['ENABLE_MODALITY_AGGREGATION'] = 'true'
    
    # Set individual client environment variables as backup
    for client_id, dataset_name in client_datasets.items():
        os.environ[f'CLIENT_{client_id}_DATASET'] = dataset_name

def setup_environment_variables(args):
    """Set up environment variables for federated learning"""
    os.environ['TASK_NAME'] = args.dataset
    os.environ['NUM_CLIENTS'] = str(args.clients)
    os.environ['NUM_TRAINING_ROUNDS'] = str(args.rounds)
    os.environ['LOCAL_EPOCHS'] = str(args.local_epochs)
    os.environ['VALIDATE_ENABLED'] = str(args.validate and not args.no_validate)
    os.environ['VALIDATION_FREQUENCY'] = str(args.validation_frequency)
    os.environ['OUTPUT_DIR'] = args.output_dir
    os.environ['SAVE_FREQUENCY'] = str(args.save_frequency)
    
    # Modality-aware aggregation settings
    if args.enable_modality_aggregation:
        os.environ['ENABLE_MODALITY_AGGREGATION'] = 'true'
        if args.modality_weights:
            os.environ['MODALITY_WEIGHTS'] = args.modality_weights

def create_federation_config(args) -> Dict[str, Any]:
    """Create federation configuration for pyproject.toml"""
    config = {
        'num-server-rounds': args.rounds + 2,  # +2 for preprocessing and initialization
        'fraction-fit': 1.0,
        'fraction-evaluate': 0.0,
        'superlink-host': args.superlink_host,
        'superlink-port': args.superlink_port,
        'num-supernodes': args.clients,
        'enable-modality-aggregation': args.enable_modality_aggregation
    }
    
    if args.modality_weights:
        config['modality-weights'] = args.modality_weights
        
    return config

def start_superlink(args) -> subprocess.Popen:
    """Start the Flower SuperLink server"""
    cmd = [
        'flower-superlink',
        '--insecure'
    ]
    
    print(f"🚀 Starting SuperLink: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def start_supernode(args, node_id: int = None) -> subprocess.Popen:
    """Start a Flower SuperNode client"""
    if node_id is None:
        node_id = args.node_id
        
    # Use different ClientApp API addresses for each SuperNode
    clientapp_port = 9094 + node_id
    
    cmd = [
        'flower-supernode',
        '--insecure',
        '--superlink', f'{args.superlink_host}:9092',
        '--clientappio-api-address', f'{args.superlink_host}:{clientapp_port}',
        '--node-config', f'partition-id={args.partition_id + node_id}'
    ]
    
    print(f"🔗 Starting SuperNode {node_id}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=dict(os.environ, **{
        'CLIENT_ID': str(node_id),
        'PARTITION_ID': str(args.partition_id + node_id)
    }))

def run_federation(args):
    """Run the complete federated learning with SuperLink and SuperNodes"""
    cmd = [
        'flwr', 'run', '.',
        'deployment'  # Use the deployment federation
    ]
    
    print(f"▶️  Running Federation: {' '.join(cmd)}")
    return subprocess.run(cmd)

def cleanup_processes(processes: List[subprocess.Popen]):
    """Clean up running processes"""
    print("\n🧹 Cleaning up processes...")
    for i, process in enumerate(processes):
        if process and process.poll() is None:
            print(f"Terminating process {i}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing process {i}...")
                process.kill()

def signal_handler(signum, frame, processes):
    """Handle interrupt signals"""
    print(f"\n📡 Received signal {signum}")
    cleanup_processes(processes)
    sys.exit(0)

def main():
    print("=== Flower SuperNode/SuperLink based nnUNet Federated Learning ===")
    
    # Parse arguments
    args = parse_arguments()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Define paths - get from environment variable or use fallback
    preproc_root = os.environ.get('nnUNet_preprocessed', os.path.expanduser('~/nnUNet_preprocessed'))
    
    # Handle list datasets option
    if args.list_datasets:
        print("\nAvailable datasets:")
        datasets = list_available_datasets(preproc_root)
        if datasets:
            for i, dataset in enumerate(datasets, 1):
                print(f"  {i}. {dataset}")
        else:
            print("  No datasets found in", preproc_root)
        return
    
    # Handle multi-dataset configuration
    client_datasets = {}
    if args.client_datasets:
        # Multi-dataset federation
        client_datasets = parse_client_datasets(args.client_datasets)
        if not client_datasets:
            print("❌ Invalid client-datasets configuration")
            return
        
        # Validate multi-dataset setup
        if not validate_multi_dataset_setup(client_datasets, preproc_root):
            if not args.validate_datasets:
                print("❌ Multi-dataset validation failed. Use --validate-datasets to see details.")
                return
        
        # Set up multi-dataset environment
        setup_multi_dataset_environment(client_datasets, args)
        
    else:
        # Single dataset federation (traditional)
        if not validate_dataset(args.dataset, preproc_root):
            available = list_available_datasets(preproc_root)
            print(f"❌ Dataset '{args.dataset}' not found or invalid.")
            print(f"Available datasets: {', '.join(available)}")
            return
        
        # Create single-dataset mapping for all clients
        client_datasets = {str(i): args.dataset for i in range(args.clients)}
        
        # Set up environment variables
        setup_environment_variables(args)
    
    # Display configuration
    if args.client_datasets:
        unique_datasets = set(client_datasets.values())
        print(f"📊 Federation Type: Multi-Dataset ({len(unique_datasets)} datasets)")
        print(f"🏥 Client-Dataset Mapping:")
        for client_id, dataset_name in client_datasets.items():
            print(f"   Client {client_id}: {dataset_name}")
    else:
        print(f"📊 Federation Type: Single-Dataset ({args.dataset})")
    
    print(f"👥 Clients: {args.clients}")
    print(f"🔄 Rounds: {args.rounds}")
    print(f"📈 Local epochs: {args.local_epochs}")
    print(f"🌐 SuperLink: {args.superlink_host}:{args.superlink_port}")
    print(f"🧠 Modality-aware aggregation: {'enabled' if args.enable_modality_aggregation else 'disabled'}")
    print(f"✅ Validation: {'enabled' if args.validate and not args.no_validate else 'disabled'}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    processes = []
    
    try:
        if args.mode == 'superlink':
            # Start SuperLink only
            superlink_proc = start_superlink(args)
            processes.append(superlink_proc)
            print("SuperLink started. Waiting...")
            superlink_proc.wait()
            
        elif args.mode == 'supernode':
            # Start SuperNode only
            print("⏳ Waiting 2 seconds for SuperLink to be ready...")
            time.sleep(2)
            
            supernode_proc = start_supernode(args)
            processes.append(supernode_proc)
            print("SuperNode started. Waiting...")
            supernode_proc.wait()
            
        elif args.mode == 'run':
            # Full deployment: Start SuperLink, SuperNodes, and run federation
            
            # Set up signal handlers for cleanup
            signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, processes))
            signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, processes))
            
            # Start SuperLink
            superlink_proc = start_superlink(args)
            processes.append(superlink_proc)
            
            # Wait for SuperLink to be ready
            print("⏳ Waiting 3 seconds for SuperLink to be ready...")
            time.sleep(3)
            
            # Start SuperNodes
            supernode_processes = []
            for i in range(args.clients):
                supernode_proc = start_supernode(args, node_id=i)
                supernode_processes.append(supernode_proc)
                processes.append(supernode_proc)
                time.sleep(1)  # Stagger SuperNode startup
            
            # Wait for SuperNodes to be ready
            print("⏳ Waiting 5 seconds for SuperNodes to be ready...")
            time.sleep(5)
            
            # Run the federation
            print("🚀 Starting federated learning...")
            result = run_federation(args)
            
            print(f"\n🎉 Federated learning completed with exit code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_processes(processes)

if __name__ == "__main__":
    main()