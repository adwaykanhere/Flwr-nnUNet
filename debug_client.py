#!/usr/bin/env python3

import os
import json
import sys

def test_data_files():
    """Test if required data files exist and are valid"""
    
    task_name = os.environ.get("TASK_NAME", "Dataset005_Prostate")
    preproc_root = os.environ.get("nnUNet_preprocessed", "/mnt/c/Users/adway/Documents/nnUNet_preprocessed")
    
    print(f"Testing data files for task: {task_name}")
    print(f"Preprocessed root: {preproc_root}")
    
    # Check paths
    plans_path = os.path.join(preproc_root, task_name, "nnUNetPlans.json")
    dataset_json = os.path.join(preproc_root, task_name, "dataset.json")
    dataset_fp = os.path.join(preproc_root, task_name, "dataset_fingerprint.json")
    
    print(f"\nChecking files:")
    print(f"Plans: {plans_path}")
    print(f"  Exists: {os.path.exists(plans_path)}")
    
    print(f"Dataset JSON: {dataset_json}")
    print(f"  Exists: {os.path.exists(dataset_json)}")
    
    print(f"Fingerprint: {dataset_fp}")
    print(f"  Exists: {os.path.exists(dataset_fp)}")
    
    # Try to load files
    if os.path.exists(plans_path):
        try:
            with open(plans_path, 'r') as f:
                plans = json.load(f)
            print(f"  Plans loaded successfully, keys: {list(plans.keys())}")
        except Exception as e:
            print(f"  Error loading plans: {e}")
    
    if os.path.exists(dataset_json):
        try:
            with open(dataset_json, 'r') as f:
                dataset = json.load(f)
            print(f"  Dataset loaded successfully, training cases: {len(dataset.get('training', []))}")
        except Exception as e:
            print(f"  Error loading dataset: {e}")
    
    if os.path.exists(dataset_fp):
        try:
            with open(dataset_fp, 'r') as f:
                fp = json.load(f)
            print(f"  Fingerprint loaded successfully, keys: {list(fp.keys())}")
        except Exception as e:
            print(f"  Error loading fingerprint: {e}")

def test_simple_client_creation():
    """Test creating client without full initialization"""
    print("\n" + "="*50)
    print("Testing simple client creation...")
    
    try:
        from task import FedNnUNetTrainer
        print("✓ FedNnUNetTrainer import successful")
        
        # Test creating trainer without full initialization
        task_name = os.environ.get("TASK_NAME", "Dataset005_Prostate")
        preproc_root = os.environ.get("nnUNet_preprocessed", "/mnt/c/Users/adway/Documents/nnUNet_preprocessed")
        plans_path = os.path.join(preproc_root, task_name, "nnUNetPlans.json")
        dataset_json = os.path.join(preproc_root, task_name, "dataset.json")
        out_root = os.environ.get("OUTPUT_ROOT", "/mnt/c/Users/adway/Documents/nnunet_output")
        
        if os.path.exists(plans_path) and os.path.exists(dataset_json):
            print("✓ Required files exist, testing trainer creation...")
            
            # Create trainer but don't initialize yet
            trainer = FedNnUNetTrainer(
                plans=plans_path,
                configuration="3d_fullres",
                fold=0,
                dataset_json=dataset_json,
                output_folder=os.path.join(out_root, "test_client"),
                max_num_epochs=5,
            )
            print("✓ FedNnUNetTrainer created successfully (not initialized)")
            
            # Test initialization separately
            print("Testing trainer initialization...")
            trainer.initialize()
            print("✓ Trainer initialized successfully")
            
        else:
            print("✗ Required files missing, cannot test trainer creation")
            
    except Exception as e:
        print(f"✗ Error in client creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*50)
    print("DEBUG: Testing nnUNet client setup")
    print("="*50)
    
    test_data_files()
    test_simple_client_creation()
    
    print("\n" + "="*50)
    print("Debug complete")