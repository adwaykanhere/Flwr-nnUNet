#!/usr/bin/env python3
"""
Simple test script to verify nnUNet trainer initialization works.
"""

import os
import sys
import json
import torch


def get_nnunet_preprocessed_path():
    """Get nnUNet preprocessed path from environment variable or user input."""
    # Try environment variable first
    nnunet_path = os.environ.get("nnUNet_preprocessed")
    if nnunet_path and os.path.exists(nnunet_path):
        print(f"Using nnUNet preprocessed path from environment: {nnunet_path}")
        return nnunet_path
    
    # Ask user for path
    while True:
        nnunet_path = input("Please enter the path to your nnUNet_preprocessed directory: ").strip()
        if not nnunet_path:
            print("Path cannot be empty. Please try again.")
            continue
        
        # Expand user home directory if needed
        nnunet_path = os.path.expanduser(nnunet_path)
        
        if os.path.exists(nnunet_path):
            print(f"Using nnUNet preprocessed path: {nnunet_path}")
            return nnunet_path
        else:
            print(f"Directory does not exist: {nnunet_path}")
            print("Please check the path and try again.")

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_trainer_basic():
    """Test basic trainer initialization"""
    print("Testing basic trainer initialization...")
    
    from task import FedNnUNetTrainer
    
    # Get configurable paths
    preproc_root = get_nnunet_preprocessed_path()
    task_name = "Dataset005_Prostate"
    plans_path = os.path.join(preproc_root, task_name, "nnUNetPlans.json")
    dataset_json = os.path.join(preproc_root, task_name, "dataset.json")
    
    # Check files exist
    if not os.path.exists(plans_path):
        print(f"ERROR: Plans file not found: {plans_path}")
        return False
        
    if not os.path.exists(dataset_json):
        print(f"ERROR: Dataset file not found: {dataset_json}")
        return False
    
    # Load JSON files
    with open(plans_path, "r") as f:
        plans_dict = json.load(f)
    with open(dataset_json, "r") as f:
        dataset_dict = json.load(f)
    
    print("Creating trainer...")
    trainer = FedNnUNetTrainer(
        plans=plans_dict,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_dict,
        device=torch.device("cpu"),
    )
    
    print("Initializing trainer...")
    try:
        trainer.initialize()
        print("✓ Trainer initialized successfully")
        
        # Test get_weights
        weights = trainer.get_weights()
        print(f"✓ Model has {len(weights)} parameter tensors")
        
        # Test set_weights
        trainer.set_weights(weights)
        print("✓ Weight setting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """Test dataloader creation"""
    print("\nTesting dataloader creation...")
    
    from task import FedNnUNetTrainer
    
    # Get configurable paths
    preproc_root = get_nnunet_preprocessed_path()
    task_name = "Dataset005_Prostate"
    plans_path = os.path.join(preproc_root, task_name, "nnUNetPlans.json")
    dataset_json = os.path.join(preproc_root, task_name, "dataset.json")
    
    # Load JSON files
    with open(plans_path, "r") as f:
        plans_dict = json.load(f)
    with open(dataset_json, "r") as f:
        dataset_dict = json.load(f)
    
    trainer = FedNnUNetTrainer(
        plans=plans_dict,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_dict,
        device=torch.device("cpu"),
    )
    
    try:
        trainer.initialize()
        print("✓ Dataloaders created successfully")
        
        # Test getting one batch
        print("Testing batch retrieval...")
        batch_count = 0
        for batch_data in trainer.dataloader_train:
            if batch_data is None:
                print("✗ Got None batch data")
                break
            print(f"✓ Got batch {batch_count + 1}")
            print(f"  Batch type: {type(batch_data)}")
            print(f"  Batch keys: {list(batch_data.keys()) if isinstance(batch_data, dict) else 'not a dict'}")
            if isinstance(batch_data, dict):
                if 'data' in batch_data:
                    print(f"  Data shape: {batch_data['data'].shape}")
                if 'target' in batch_data:
                    target = batch_data['target']
                    if isinstance(target, list):
                        print(f"  Target shape: [deep supervision: {len(target)} levels]")
                    else:
                        print(f"  Target shape: {target.shape}")
            batch_count += 1
            if batch_count >= 1:  # Just test one batch
                break
                
        print("✓ Batch retrieval works")
        return True
        
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running nnUNet trainer tests...")
    
    success1 = test_trainer_basic()
    success2 = test_dataloader()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
