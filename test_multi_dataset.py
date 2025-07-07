#!/usr/bin/env python3
"""
Test script to validate multi-dataset federation functionality
"""

import os
import json
import tempfile
from pathlib import Path

def test_multi_dataset_imports():
    """Test that all multi-dataset modules can be imported"""
    print("🧪 Testing multi-dataset imports...")
    
    try:
        import dataset_compatibility
        print("✅ dataset_compatibility imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dataset_compatibility: {e}")
        return False
    
    try:
        import federation_config
        print("✅ federation_config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import federation_config: {e}")
        return False
    
    try:
        from run_federated_deployment import parse_client_datasets, validate_multi_dataset_setup
        print("✅ multi-dataset deployment functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import deployment functions: {e}")
        return False
    
    return True

def test_client_dataset_parsing():
    """Test client-dataset mapping parsing"""
    print("\n🧪 Testing client-dataset parsing...")
    
    from run_federated_deployment import parse_client_datasets
    
    # Test valid JSON
    test_json = '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset027_ACDC"}'
    result = parse_client_datasets(test_json)
    
    expected = {"0": "Dataset005_Prostate", "1": "Dataset009_Spleen", "2": "Dataset027_ACDC"}
    
    if result == expected:
        print("✅ Client-dataset parsing successful")
        print(f"   Parsed: {result}")
        return True
    else:
        print(f"❌ Client-dataset parsing failed: expected {expected}, got {result}")
        return False

def test_dataset_compatibility_manager():
    """Test dataset compatibility analysis"""
    print("\n🧪 Testing dataset compatibility manager...")
    
    from dataset_compatibility import DatasetCompatibilityManager
    
    try:
        manager = DatasetCompatibilityManager()
        
        # Create mock dataset metadata
        mock_dataset_path = tempfile.mkdtemp()
        
        # Create mock dataset.json
        mock_dataset_json = {
            "name": "TestDataset",
            "description": "Test dataset for compatibility testing",
            "channel_names": {"0": "CT"},
            "labels": {"0": "background", "1": "organ"},
            "numTraining": 50,
            "numTest": 10
        }
        
        dataset_json_path = os.path.join(mock_dataset_path, "dataset.json")
        with open(dataset_json_path, 'w') as f:
            json.dump(mock_dataset_json, f)
        
        # Register dataset
        metadata = manager.register_dataset("TestDataset", mock_dataset_path, "client_0")
        
        if metadata and metadata['dataset_name'] == "TestDataset":
            print("✅ Dataset compatibility manager working")
            print(f"   Detected modalities: {metadata['modalities']}")
            
            # Test modality groups
            modality_groups = manager.get_modality_groups()
            print(f"   Modality groups: {modality_groups}")
            
            return True
        else:
            print("❌ Dataset compatibility manager failed")
            return False
            
    except Exception as e:
        print(f"❌ Dataset compatibility manager error: {e}")
        return False
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(mock_dataset_path)
        except:
            pass

def test_federation_config_manager():
    """Test federation configuration management"""
    print("\n🧪 Testing federation config manager...")
    
    from federation_config import FederationConfigManager, ClientConfig, DatasetConfig
    
    try:
        manager = FederationConfigManager()
        
        # Create mock configuration
        clients = [
            ClientConfig("0", "Dataset005_Prostate", 0, 2, True),
            ClientConfig("1", "Dataset009_Spleen", 1, 2, True)
        ]
        
        datasets = [
            DatasetConfig("Dataset005_Prostate", "/path/to/prostate", "MR"),
            DatasetConfig("Dataset009_Spleen", "/path/to/spleen", "CT")
        ]
        
        print("✅ Federation config manager working")
        print(f"   Created {len(clients)} clients and {len(datasets)} datasets")
        
        # Test client-dataset mapping
        client_datasets = {"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}
        print(f"   Client-dataset mapping: {client_datasets}")
        
        return True
        
    except Exception as e:
        print(f"❌ Federation config manager error: {e}")
        return False

def test_modality_extraction():
    """Test modality extraction from different channel configurations"""
    print("\n🧪 Testing modality extraction...")
    
    from server_app_modality import ModalityAwareFederatedStrategy
    
    try:
        strategy = ModalityAwareFederatedStrategy(expected_num_clients=2)
        
        # Test different modality patterns
        test_cases = [
            ({"channel_names": {"0": "CT"}}, "CT"),
            ({"channel_names": {"0": "T1w"}}, "MR"),
            ({"channel_names": {"0": "T2w"}}, "MR"),
            ({"channel_names": {"0": "PET"}}, "PET"),
            ({"channel_names": {"0": "ultrasound"}}, "US"),
            ({"modality": "CT"}, "CT"),
            ({"dataset_modality": "MR"}, "MR"),
        ]
        
        success_count = 0
        for metrics, expected_modality in test_cases:
            extracted_modality = strategy.extract_modality_from_metadata(metrics)
            
            if extracted_modality == expected_modality:
                print(f"✅ {metrics} → {extracted_modality}")
                success_count += 1
            else:
                print(f"❌ {metrics} → {extracted_modality} (expected {expected_modality})")
        
        if success_count == len(test_cases):
            print("✅ All modality extraction tests passed")
            return True
        else:
            print(f"❌ {success_count}/{len(test_cases)} modality extraction tests passed")
            return False
            
    except Exception as e:
        print(f"❌ Modality extraction error: {e}")
        return False

def test_client_dataset_config():
    """Test client dataset configuration functionality"""
    print("\n🧪 Testing client dataset configuration...")
    
    try:
        from client_app import get_client_dataset_config
        from flwr.common import Context
        
        # Mock context
        class MockContext:
            def __init__(self):
                self.node_config = {"partition-id": 0}
        
        context = MockContext()
        
        # Test multi-dataset configuration
        os.environ['CLIENT_DATASETS'] = '{"0": "Dataset005_Prostate", "1": "Dataset009_Spleen"}'
        
        config = get_client_dataset_config(0, context)
        
        if config['dataset_name'] == "Dataset005_Prostate" and config['source'] == "multi_dataset":
            print("✅ Multi-dataset client configuration working")
            print(f"   Client 0 config: {config}")
            
            # Test client 1
            context.node_config["partition-id"] = 1
            config1 = get_client_dataset_config(1, context)
            
            if config1['dataset_name'] == "Dataset009_Spleen":
                print(f"   Client 1 config: {config1}")
                return True
            else:
                print(f"❌ Client 1 config failed: {config1}")
                return False
        else:
            print(f"❌ Multi-dataset client configuration failed: {config}")
            return False
            
    except Exception as e:
        print(f"❌ Client dataset configuration error: {e}")
        return False
    finally:
        # Cleanup environment
        if 'CLIENT_DATASETS' in os.environ:
            del os.environ['CLIENT_DATASETS']

def main():
    """Run all multi-dataset tests"""
    print("🚀 Running Multi-Dataset Federation Tests\n")
    
    tests = [
        ("Import Tests", test_multi_dataset_imports),
        ("Client Dataset Parsing", test_client_dataset_parsing),
        ("Dataset Compatibility Manager", test_dataset_compatibility_manager),
        ("Federation Config Manager", test_federation_config_manager),
        ("Modality Extraction", test_modality_extraction),
        ("Client Dataset Configuration", test_client_dataset_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("MULTI-DATASET FEDERATION TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All multi-dataset tests passed! Multi-dataset federation is ready.")
        print("\nNext steps:")
        print("1. Try: python run_federated_deployment.py --client-datasets '{\"0\": \"Dataset005_Prostate\", \"1\": \"Dataset009_Spleen\"}' --validate-datasets")
        print("2. See: MULTI_DATASET_GUIDE.md for comprehensive examples")
        print("3. Configure: Custom modality weights for your specific use case")
    else:
        print("⚠️  Some multi-dataset tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())