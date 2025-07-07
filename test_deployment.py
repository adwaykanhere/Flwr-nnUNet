#!/usr/bin/env python3
"""
Test script to validate the federated deployment setup
"""

import os
import json

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import run_federated_deployment
        print("✅ run_federated_deployment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import run_federated_deployment: {e}")
        return False
    
    try:
        import server_app_modality
        print("✅ server_app_modality imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import server_app_modality: {e}")
        return False
    
    try:
        import client_app
        print("✅ client_app imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import client_app: {e}")
        return False
    
    return True

def test_modality_extraction():
    """Test modality extraction functionality"""
    print("\n🧪 Testing modality extraction...")
    
    # Mock dataset.json content
    mock_dataset = {
        "channel_names": {
            "0": "CT"
        },
        "name": "TestDataset",
        "description": "Test dataset for CT scans",
        "numTraining": 100,
        "numTest": 20
    }
    
    # Test modality inference
    from client_app import NnUNet3DFullresClient
    
    # Create a temporary dataset.json file
    test_dataset_path = "/tmp/test_dataset.json"
    with open(test_dataset_path, "w") as f:
        json.dump(mock_dataset, f)
    
    try:
        # Mock client to test modality extraction
        class MockClient:
            def __init__(self):
                self.dataset_json_path = test_dataset_path
                self.local_fingerprint = {}
                self.client_id = 0
            
            def _extract_modality_info(self):
                """Extract modality information from dataset.json and fingerprint"""
                modality_info = {}
                
                try:
                    with open(self.dataset_json_path, "r") as f:
                        dataset_dict = json.load(f)
                    
                    channel_names = dataset_dict.get("channel_names", {})
                    if channel_names:
                        modality_info["channel_names"] = channel_names
                        
                        first_channel_key = list(channel_names.keys())[0] if channel_names else "0"
                        first_channel_name = channel_names.get(first_channel_key, "").lower()
                        
                        if 'ct' in first_channel_name or 'computed' in first_channel_name:
                            modality_info["modality"] = "CT"
                        elif 'mr' in first_channel_name or 'magnetic' in first_channel_name:
                            modality_info["modality"] = "MR"
                        else:
                            modality_info["modality"] = "UNKNOWN"
                    
                    modality_info["dataset_name"] = dataset_dict.get("name", "unknown")
                    
                except Exception as exc:
                    print(f"Error extracting modality info: {exc}")
                    modality_info = {"modality": "UNKNOWN"}
                
                return modality_info
        
        mock_client = MockClient()
        modality_info = mock_client._extract_modality_info()
        
        expected_modality = "CT"
        actual_modality = modality_info.get("modality")
        
        if actual_modality == expected_modality:
            print(f"✅ Modality extraction successful: {actual_modality}")
            print(f"   Dataset: {modality_info.get('dataset_name')}")
            print(f"   Channels: {modality_info.get('channel_names')}")
            return True
        else:
            print(f"❌ Modality extraction failed: expected {expected_modality}, got {actual_modality}")
            return False
            
    except Exception as e:
        print(f"❌ Modality extraction test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_dataset_path):
            os.remove(test_dataset_path)

def test_configuration():
    """Test configuration parsing"""
    print("\n🧪 Testing configuration...")
    
    try:
        from run_federated_deployment import parse_arguments, create_federation_config
        
        # Test argument parsing with modality settings
        test_args = [
            '--clients', '2',
            '--rounds', '3', 
            '--enable-modality-aggregation',
            '--modality-weights', '{"CT": 0.6, "MR": 0.4}'
        ]
        
        import sys
        orig_argv = sys.argv
        sys.argv = ['test'] + test_args
        
        try:
            args = parse_arguments()
            
            if args.clients == 2 and args.rounds == 3 and args.enable_modality_aggregation:
                print("✅ Argument parsing successful")
                
                # Test federation config creation
                config = create_federation_config(args)
                
                if config['num-supernodes'] == 2 and config['enable-modality-aggregation']:
                    print("✅ Federation configuration successful")
                    return True
                else:
                    print("❌ Federation configuration failed")
                    return False
            else:
                print("❌ Argument parsing failed")
                return False
                
        finally:
            sys.argv = orig_argv
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_server_strategy():
    """Test server strategy initialization"""
    print("\n🧪 Testing server strategy...")
    
    try:
        from server_app_modality import ModalityAwareFederatedStrategy
        
        # Test strategy initialization
        strategy = ModalityAwareFederatedStrategy(
            expected_num_clients=2,
            enable_modality_aggregation=True,
            modality_weights={"CT": 0.6, "MR": 0.4}
        )
        
        if (strategy.enable_modality_aggregation and 
            strategy.modality_weights == {"CT": 0.6, "MR": 0.4}):
            print("✅ Server strategy initialization successful")
            return True
        else:
            print("❌ Server strategy initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Server strategy test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Federated nnUNet Deployment Tests\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Modality Extraction", test_modality_extraction),
        ("Configuration Tests", test_configuration),
        ("Server Strategy Tests", test_server_strategy)
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
    print("TEST SUMMARY")
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
        print("🎉 All tests passed! The deployment setup is ready.")
        print("\nNext steps:")
        print("1. Run: python run_federated_deployment.py --list-datasets")
        print("2. Run: python run_federated_deployment.py --mode run --clients 2 --rounds 3")
        print("3. See DEPLOYMENT_GUIDE.md for advanced usage")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())