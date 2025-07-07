# federation_config.py

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """Configuration for a single client"""
    client_id: str
    dataset: str
    partition_id: Optional[int] = None
    local_epochs: Optional[int] = None
    validation_enabled: Optional[bool] = None
    output_dir: Optional[str] = None
    gpu_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    path: str
    modality: Optional[str] = None
    description: Optional[str] = None
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AggregationConfig:
    """Configuration for aggregation strategy"""
    strategy: str = "traditional"  # traditional, modality_aware, multi_dataset
    enable_modality_aggregation: bool = False
    modality_weights: Optional[Dict[str, float]] = None
    dataset_weights: Optional[Dict[str, float]] = None
    dataset_modality_weights: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    mode: str = "run"  # run, superlink, supernode
    superlink_host: str = "127.0.0.1"
    superlink_port: int = 9091
    insecure: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    rounds: int = 3
    local_epochs: int = 2
    validation_enabled: bool = True
    validation_frequency: int = 1
    save_frequency: int = 1
    output_dir: str = "federated_models"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FederationConfig:
    """Complete federation configuration"""
    name: str
    description: str
    clients: List[ClientConfig]
    datasets: List[DatasetConfig]
    aggregation: AggregationConfig
    deployment: DeploymentConfig
    training: TrainingConfig
    preprocessed_root: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'clients': [c.to_dict() for c in self.clients],
            'datasets': [d.to_dict() for d in self.datasets],
            'aggregation': self.aggregation.to_dict(),
            'deployment': self.deployment.to_dict(),
            'training': self.training.to_dict(),
            'preprocessed_root': self.preprocessed_root
        }

class FederationConfigManager:
    """Manager for federation configurations"""
    
    def __init__(self):
        self.config: Optional[FederationConfig] = None
        
    def create_config_from_args(self, args, preproc_root: str) -> FederationConfig:
        """Create configuration from command line arguments"""
        
        # Create client configurations
        clients = []
        if hasattr(args, 'client_datasets') and args.client_datasets:
            # Multi-dataset configuration
            from run_federated_deployment import parse_client_datasets
            client_datasets = parse_client_datasets(args.client_datasets)
            
            for client_id, dataset_name in client_datasets.items():
                client_config = ClientConfig(
                    client_id=client_id,
                    dataset=dataset_name,
                    partition_id=int(client_id),
                    local_epochs=args.local_epochs,
                    validation_enabled=args.validate and not args.no_validate,
                    output_dir=args.output_dir,
                    gpu_id=args.gpu
                )
                clients.append(client_config)
        else:
            # Single dataset configuration
            for i in range(args.clients):
                client_config = ClientConfig(
                    client_id=str(i),
                    dataset=args.dataset,
                    partition_id=i,
                    local_epochs=args.local_epochs,
                    validation_enabled=args.validate and not args.no_validate,
                    output_dir=args.output_dir,
                    gpu_id=args.gpu
                )
                clients.append(client_config)
        
        # Create dataset configurations
        datasets = []
        unique_datasets = set(client.dataset for client in clients)
        
        for dataset_name in unique_datasets:
            dataset_path = os.path.join(preproc_root, dataset_name)
            dataset_config = DatasetConfig(
                name=dataset_name,
                path=dataset_path,
                description=f"Dataset {dataset_name}",
                priority=1.0
            )
            datasets.append(dataset_config)
        
        # Create aggregation configuration
        modality_weights = None
        if hasattr(args, 'modality_weights') and args.modality_weights:
            try:
                modality_weights = json.loads(args.modality_weights)
            except json.JSONDecodeError:
                logger.warning("Invalid modality weights JSON, ignoring")
        
        aggregation_config = AggregationConfig(
            strategy="multi_dataset" if len(unique_datasets) > 1 else "modality_aware" if args.enable_modality_aggregation else "traditional",
            enable_modality_aggregation=args.enable_modality_aggregation,
            modality_weights=modality_weights
        )
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            mode=args.mode,
            superlink_host=args.superlink_host,
            superlink_port=args.superlink_port,
            insecure=args.insecure
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            validation_enabled=args.validate and not args.no_validate,
            validation_frequency=getattr(args, 'validation_frequency', 1),
            save_frequency=getattr(args, 'save_frequency', 1),
            output_dir=args.output_dir
        )
        
        # Create complete configuration
        config = FederationConfig(
            name=f"federation_{len(clients)}clients_{len(unique_datasets)}datasets",
            description=f"Federation with {len(clients)} clients and {len(unique_datasets)} datasets",
            clients=clients,
            datasets=datasets,
            aggregation=aggregation_config,
            deployment=deployment_config,
            training=training_config,
            preprocessed_root=preproc_root
        )
        
        self.config = config
        return config
    
    def load_config_from_file(self, config_path: str) -> FederationConfig:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Parse configuration data
        config = self._parse_config_data(config_data)
        self.config = config
        return config
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> FederationConfig:
        """Parse configuration data from dictionary"""
        
        # Parse clients
        clients = []
        for client_data in config_data.get('clients', []):
            client_config = ClientConfig(**client_data)
            clients.append(client_config)
        
        # Parse datasets
        datasets = []
        for dataset_data in config_data.get('datasets', []):
            dataset_config = DatasetConfig(**dataset_data)
            datasets.append(dataset_config)
        
        # Parse aggregation config
        aggregation_data = config_data.get('aggregation', {})
        aggregation_config = AggregationConfig(**aggregation_data)
        
        # Parse deployment config
        deployment_data = config_data.get('deployment', {})
        deployment_config = DeploymentConfig(**deployment_data)
        
        # Parse training config
        training_data = config_data.get('training', {})
        training_config = TrainingConfig(**training_data)
        
        # Create complete configuration
        config = FederationConfig(
            name=config_data.get('name', 'unnamed_federation'),
            description=config_data.get('description', ''),
            clients=clients,
            datasets=datasets,
            aggregation=aggregation_config,
            deployment=deployment_config,
            training=training_config,
            preprocessed_root=config_data.get('preprocessed_root', '')
        )
        
        return config
    
    def save_config_to_file(self, config_path: str, config: Optional[FederationConfig] = None) -> None:
        """Save configuration to YAML or JSON file"""
        if config is None:
            config = self.config
        
        if config is None:
            raise ValueError("No configuration to save")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = config.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def validate_config(self, config: Optional[FederationConfig] = None) -> Dict[str, Any]:
        """Validate the configuration"""
        if config is None:
            config = self.config
        
        if config is None:
            return {"valid": False, "errors": ["No configuration loaded"]}
        
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Validate clients
        if not config.clients:
            validation_report["valid"] = False
            validation_report["errors"].append("No clients configured")
        
        client_ids = [c.client_id for c in config.clients]
        if len(client_ids) != len(set(client_ids)):
            validation_report["valid"] = False
            validation_report["errors"].append("Duplicate client IDs found")
        
        # Validate datasets
        if not config.datasets:
            validation_report["valid"] = False
            validation_report["errors"].append("No datasets configured")
        
        # Check dataset paths
        for dataset in config.datasets:
            dataset_path = Path(dataset.path)
            if not dataset_path.exists():
                validation_report["warnings"].append(f"Dataset path not found: {dataset.path}")
        
        # Validate client-dataset mapping
        configured_datasets = {d.name for d in config.datasets}
        client_datasets = {c.dataset for c in config.clients}
        missing_datasets = client_datasets - configured_datasets
        
        if missing_datasets:
            validation_report["valid"] = False
            validation_report["errors"].append(f"Clients reference unconfigured datasets: {missing_datasets}")
        
        # Validate aggregation configuration
        if config.aggregation.enable_modality_aggregation and not config.aggregation.modality_weights:
            validation_report["recommendations"].append("Consider specifying modality weights for better aggregation")
        
        # Multi-dataset recommendations
        unique_datasets = len(set(c.dataset for c in config.clients))
        if unique_datasets > 1:
            if not config.aggregation.enable_modality_aggregation:
                validation_report["recommendations"].append("Enable modality aggregation for multi-dataset federation")
            
            if config.aggregation.strategy == "traditional":
                validation_report["recommendations"].append("Consider using multi_dataset aggregation strategy")
        
        return validation_report
    
    def get_client_dataset_mapping(self, config: Optional[FederationConfig] = None) -> Dict[str, str]:
        """Get client-dataset mapping"""
        if config is None:
            config = self.config
        
        if config is None:
            return {}
        
        return {client.client_id: client.dataset for client in config.clients}
    
    def get_environment_variables(self, config: Optional[FederationConfig] = None) -> Dict[str, str]:
        """Get environment variables for the configuration"""
        if config is None:
            config = self.config
        
        if config is None:
            return {}
        
        env_vars = {}
        
        # Client-dataset mapping
        client_datasets = self.get_client_dataset_mapping(config)
        env_vars['CLIENT_DATASETS'] = json.dumps(client_datasets)
        
        # Training configuration
        env_vars['NUM_CLIENTS'] = str(len(config.clients))
        env_vars['NUM_TRAINING_ROUNDS'] = str(config.training.rounds)
        env_vars['LOCAL_EPOCHS'] = str(config.training.local_epochs)
        env_vars['VALIDATE_ENABLED'] = str(config.training.validation_enabled).lower()
        env_vars['OUTPUT_DIR'] = config.training.output_dir
        
        # Aggregation configuration
        if config.aggregation.enable_modality_aggregation:
            env_vars['ENABLE_MODALITY_AGGREGATION'] = 'true'
            
            if config.aggregation.modality_weights:
                env_vars['MODALITY_WEIGHTS'] = json.dumps(config.aggregation.modality_weights)
        
        # Dataset root
        env_vars['nnUNet_preprocessed'] = config.preprocessed_root
        
        return env_vars


def create_example_configs():
    """Create example configuration files"""
    
    # Single-dataset configuration
    single_dataset_config = {
        "name": "single_dataset_federation",
        "description": "Simple federation with single dataset",
        "preprocessed_root": "/path/to/nnUNet_preprocessed",
        "clients": [
            {
                "client_id": "0",
                "dataset": "Dataset005_Prostate",
                "partition_id": 0,
                "local_epochs": 2,
                "validation_enabled": True,
                "gpu_id": 0
            },
            {
                "client_id": "1", 
                "dataset": "Dataset005_Prostate",
                "partition_id": 1,
                "local_epochs": 2,
                "validation_enabled": True,
                "gpu_id": 0
            }
        ],
        "datasets": [
            {
                "name": "Dataset005_Prostate",
                "path": "/path/to/nnUNet_preprocessed/Dataset005_Prostate",
                "modality": "MR",
                "description": "Prostate segmentation dataset",
                "priority": 1.0
            }
        ],
        "aggregation": {
            "strategy": "traditional",
            "enable_modality_aggregation": False
        },
        "deployment": {
            "mode": "run",
            "superlink_host": "127.0.0.1",
            "superlink_port": 9091,
            "insecure": True
        },
        "training": {
            "rounds": 5,
            "local_epochs": 2,
            "validation_enabled": True,
            "validation_frequency": 1,
            "save_frequency": 1,
            "output_dir": "federated_models"
        }
    }
    
    # Multi-dataset configuration
    multi_dataset_config = {
        "name": "multi_dataset_federation",
        "description": "Advanced federation with multiple datasets and modalities",
        "preprocessed_root": "/path/to/nnUNet_preprocessed",
        "clients": [
            {
                "client_id": "0",
                "dataset": "Dataset005_Prostate",
                "partition_id": 0,
                "local_epochs": 3,
                "validation_enabled": True,
                "gpu_id": 0
            },
            {
                "client_id": "1",
                "dataset": "Dataset009_Spleen",
                "partition_id": 1,
                "local_epochs": 3,
                "validation_enabled": True,
                "gpu_id": 0
            },
            {
                "client_id": "2",
                "dataset": "Dataset027_ACDC",
                "partition_id": 2,
                "local_epochs": 3,
                "validation_enabled": True,
                "gpu_id": 0
            }
        ],
        "datasets": [
            {
                "name": "Dataset005_Prostate",
                "path": "/path/to/nnUNet_preprocessed/Dataset005_Prostate",
                "modality": "MR",
                "description": "Prostate segmentation (MR)",
                "priority": 1.0
            },
            {
                "name": "Dataset009_Spleen",
                "path": "/path/to/nnUNet_preprocessed/Dataset009_Spleen",
                "modality": "CT",
                "description": "Spleen segmentation (CT)",
                "priority": 1.0
            },
            {
                "name": "Dataset027_ACDC",
                "path": "/path/to/nnUNet_preprocessed/Dataset027_ACDC",
                "modality": "MR",
                "description": "Cardiac segmentation (MR)",
                "priority": 0.8
            }
        ],
        "aggregation": {
            "strategy": "multi_dataset",
            "enable_modality_aggregation": True,
            "modality_weights": {
                "CT": 0.4,
                "MR": 0.6
            },
            "dataset_modality_weights": {
                "Dataset005_Prostate_MR": 0.3,
                "Dataset009_Spleen_CT": 0.4,
                "Dataset027_ACDC_MR": 0.3
            }
        },
        "deployment": {
            "mode": "run",
            "superlink_host": "127.0.0.1",
            "superlink_port": 9091,
            "insecure": True
        },
        "training": {
            "rounds": 10,
            "local_epochs": 3,
            "validation_enabled": True,
            "validation_frequency": 2,
            "save_frequency": 2,
            "output_dir": "multi_dataset_models"
        }
    }
    
    return single_dataset_config, multi_dataset_config


if __name__ == "__main__":
    # Create example configurations
    manager = FederationConfigManager()
    
    single_config, multi_config = create_example_configs()
    
    # Save example configurations
    os.makedirs("config_examples", exist_ok=True)
    
    with open("config_examples/single_dataset.yaml", 'w') as f:
        yaml.dump(single_config, f, default_flow_style=False, indent=2)
    
    with open("config_examples/multi_dataset.yaml", 'w') as f:
        yaml.dump(multi_config, f, default_flow_style=False, indent=2)
    
    print("Example configurations created:")
    print("- config_examples/single_dataset.yaml")
    print("- config_examples/multi_dataset.yaml")