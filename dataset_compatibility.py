# dataset_compatibility.py

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCompatibilityManager:
    """
    Manages cross-dataset compatibility for federated learning with nnUNet.
    Handles different datasets, label mappings, spacing harmonization, and metadata alignment.
    """
    
    def __init__(self):
        self.dataset_metadata: Dict[str, Dict] = {}
        self.label_mappings: Dict[str, Dict] = {}
        self.harmonized_labels: Optional[Dict] = None
        self.dataset_fingerprints: Dict[str, Dict] = {}
        
    def register_dataset(self, 
                        dataset_name: str, 
                        dataset_path: str, 
                        client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a dataset and extract its metadata for compatibility analysis.
        
        Args:
            dataset_name: Name of the dataset (e.g., "Dataset005_Prostate")
            dataset_path: Path to the preprocessed dataset directory
            client_id: Optional client identifier for tracking
            
        Returns:
            Dictionary containing dataset metadata
        """
        logger.info(f"Registering dataset {dataset_name} for client {client_id}")
        
        # Load dataset.json
        dataset_json_path = os.path.join(dataset_path, "dataset.json")
        plans_json_path = os.path.join(dataset_path, "nnUNetPlans.json")
        fingerprint_path = os.path.join(dataset_path, "dataset_fingerprint.json")
        
        if not os.path.exists(dataset_json_path):
            raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")
        
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
        
        # Load plans if available
        plans_json = {}
        if os.path.exists(plans_json_path):
            with open(plans_json_path, 'r') as f:
                plans_json = json.load(f)
        
        # Load fingerprint if available
        fingerprint_json = {}
        if os.path.exists(fingerprint_path):
            with open(fingerprint_path, 'r') as f:
                fingerprint_json = json.load(f)
                self.dataset_fingerprints[dataset_name] = fingerprint_json
        
        # Extract metadata
        metadata = {
            'dataset_name': dataset_name,
            'client_id': client_id,
            'dataset_path': dataset_path,
            'name': dataset_json.get('name', dataset_name),
            'description': dataset_json.get('description', ''),
            'reference': dataset_json.get('reference', ''),
            'licence': dataset_json.get('licence', ''),
            'release': dataset_json.get('release', ''),
            'numTraining': dataset_json.get('numTraining', 0),
            'numTest': dataset_json.get('numTest', 0),
            'channel_names': dataset_json.get('channel_names', {}),
            'labels': dataset_json.get('labels', {}),
            'file_ending': dataset_json.get('file_ending', '.nii.gz'),
            'overwrite_image_reader_writer': dataset_json.get('overwrite_image_reader_writer', None),
            
            # Extract modality information
            'modalities': self._extract_modalities(dataset_json.get('channel_names', {})),
            'num_modalities': len(dataset_json.get('channel_names', {})),
            
            # Extract plans information if available
            'plans_available': bool(plans_json),
            'configurations': list(plans_json.get('configurations', {}).keys()) if plans_json else [],
            
            # Extract fingerprint information
            'fingerprint_available': bool(fingerprint_json),
            'num_cases_fingerprint': len(fingerprint_json.get('spacings', [])) if fingerprint_json else 0,
            
            # Compatibility flags
            'preprocessed': os.path.exists(plans_json_path),
            'ready_for_training': os.path.exists(plans_json_path) and os.path.exists(fingerprint_path)
        }
        
        # Store metadata
        self.dataset_metadata[dataset_name] = metadata
        
        # Store label mapping
        self.label_mappings[dataset_name] = dataset_json.get('labels', {})
        
        logger.info(f"Dataset {dataset_name} registered successfully:")
        logger.info(f"  - Modalities: {metadata['modalities']}")
        logger.info(f"  - Labels: {list(metadata['labels'].keys())}")
        logger.info(f"  - Training cases: {metadata['numTraining']}")
        
        return metadata
    
    def _extract_modalities(self, channel_names: Dict[str, str]) -> List[str]:
        """Extract modality information from channel names."""
        modalities = []
        
        for channel_key, channel_name in channel_names.items():
            channel_lower = channel_name.lower()
            
            if 'ct' in channel_lower or 'computed' in channel_lower:
                modalities.append('CT')
            elif 'mr' in channel_lower or 'magnetic' in channel_lower or 't1' in channel_lower or 't2' in channel_lower:
                modalities.append('MR')
            elif 'pet' in channel_lower:
                modalities.append('PET')
            elif 'us' in channel_lower or 'ultrasound' in channel_lower:
                modalities.append('US')
            else:
                modalities.append('UNKNOWN')
        
        return list(set(modalities))  # Remove duplicates
    
    def analyze_compatibility(self) -> Dict[str, Any]:
        """
        Analyze compatibility between all registered datasets.
        
        Returns:
            Dictionary containing compatibility analysis results
        """
        if len(self.dataset_metadata) < 2:
            return {"status": "insufficient_datasets", "datasets_count": len(self.dataset_metadata)}
        
        compatibility_report = {
            "total_datasets": len(self.dataset_metadata),
            "datasets": list(self.dataset_metadata.keys()),
            "modality_analysis": self._analyze_modalities(),
            "label_analysis": self._analyze_labels(),
            "compatibility_matrix": self._create_compatibility_matrix(),
            "recommendations": self._generate_recommendations()
        }
        
        return compatibility_report
    
    def _analyze_modalities(self) -> Dict[str, Any]:
        """Analyze modality distribution across datasets."""
        all_modalities = set()
        dataset_modalities = {}
        modality_counts = {}
        
        for dataset_name, metadata in self.dataset_metadata.items():
            modalities = metadata['modalities']
            dataset_modalities[dataset_name] = modalities
            all_modalities.update(modalities)
            
            for modality in modalities:
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "unique_modalities": sorted(list(all_modalities)),
            "dataset_modalities": dataset_modalities,
            "modality_counts": modality_counts,
            "multi_modal_datasets": [k for k, v in dataset_modalities.items() if len(v) > 1],
            "homogeneous_modalities": len(all_modalities) == 1
        }
    
    def _analyze_labels(self) -> Dict[str, Any]:
        """Analyze label compatibility across datasets."""
        all_labels = set()
        dataset_labels = {}
        label_conflicts = {}
        
        for dataset_name, labels in self.label_mappings.items():
            # Convert label values to strings for comparison
            str_labels = {str(k): str(v) for k, v in labels.items()}
            dataset_labels[dataset_name] = str_labels
            all_labels.update(str_labels.keys())
            
            # Check for conflicts with existing labels
            for label_id, label_name in str_labels.items():
                if label_id in label_conflicts:
                    if label_conflicts[label_id] != label_name:
                        logger.warning(f"Label conflict: ID {label_id} maps to '{label_conflicts[label_id]}' and '{label_name}'")
                else:
                    label_conflicts[label_id] = label_name
        
        # Find common labels
        common_labels = None
        for dataset_name, labels in dataset_labels.items():
            if common_labels is None:
                common_labels = set(labels.keys())
            else:
                common_labels = common_labels.intersection(set(labels.keys()))
        
        return {
            "unique_label_ids": sorted(list(all_labels)),
            "dataset_labels": dataset_labels,
            "common_labels": sorted(list(common_labels)) if common_labels else [],
            "label_conflicts": {k: v for k, v in label_conflicts.items() if len(set(dataset_labels[d].get(k, 'MISSING') for d in dataset_labels.keys())) > 1},
            "harmonization_needed": len(label_conflicts) > 0
        }
    
    def _create_compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """Create a compatibility matrix between datasets."""
        datasets = list(self.dataset_metadata.keys())
        matrix = {}
        
        for dataset1 in datasets:
            matrix[dataset1] = {}
            for dataset2 in datasets:
                if dataset1 == dataset2:
                    matrix[dataset1][dataset2] = 1.0
                else:
                    score = self._calculate_compatibility_score(dataset1, dataset2)
                    matrix[dataset1][dataset2] = score
        
        return matrix
    
    def _calculate_compatibility_score(self, dataset1: str, dataset2: str) -> float:
        """Calculate compatibility score between two datasets."""
        meta1 = self.dataset_metadata[dataset1]
        meta2 = self.dataset_metadata[dataset2]
        
        score = 0.0
        weight_sum = 0.0
        
        # Modality compatibility (weight: 0.4)
        modality_weight = 0.4
        modality_overlap = len(set(meta1['modalities']).intersection(set(meta2['modalities'])))
        modality_union = len(set(meta1['modalities']).union(set(meta2['modalities'])))
        modality_score = modality_overlap / modality_union if modality_union > 0 else 0.0
        score += modality_score * modality_weight
        weight_sum += modality_weight
        
        # Label compatibility (weight: 0.3)
        label_weight = 0.3
        labels1 = set(self.label_mappings.get(dataset1, {}).keys())
        labels2 = set(self.label_mappings.get(dataset2, {}).keys())
        label_overlap = len(labels1.intersection(labels2))
        label_union = len(labels1.union(labels2))
        label_score = label_overlap / label_union if label_union > 0 else 0.0
        score += label_score * label_weight
        weight_sum += label_weight
        
        # Preprocessing compatibility (weight: 0.2)
        prep_weight = 0.2
        prep_score = 1.0 if (meta1['preprocessed'] and meta2['preprocessed']) else 0.5
        score += prep_score * prep_weight
        weight_sum += prep_weight
        
        # Data availability (weight: 0.1)
        data_weight = 0.1
        data_score = 1.0 if (meta1['ready_for_training'] and meta2['ready_for_training']) else 0.5
        score += data_score * data_weight
        weight_sum += data_weight
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for federated learning setup."""
        recommendations = []
        
        modality_analysis = self._analyze_modalities()
        label_analysis = self._analyze_labels()
        
        # Modality recommendations
        if modality_analysis['homogeneous_modalities']:
            recommendations.append("✅ All datasets have the same modality - traditional FedAvg recommended")
        else:
            recommendations.append(f"🔄 Multiple modalities detected: {modality_analysis['unique_modalities']} - use modality-aware aggregation")
            if len(modality_analysis['multi_modal_datasets']) > 0:
                recommendations.append(f"⚠️  Multi-modal datasets detected: {modality_analysis['multi_modal_datasets']} - may need special handling")
        
        # Label recommendations
        if len(label_analysis['common_labels']) > 0:
            recommendations.append(f"✅ Common labels found: {label_analysis['common_labels']} - direct aggregation possible")
        else:
            recommendations.append("⚠️  No common labels - consider label harmonization or task-specific aggregation")
        
        if label_analysis['label_conflicts']:
            recommendations.append(f"❌ Label conflicts detected: {list(label_analysis['label_conflicts'].keys())} - harmonization required")
        
        # Dataset-specific recommendations
        not_ready = [name for name, meta in self.dataset_metadata.items() if not meta['ready_for_training']]
        if not_ready:
            recommendations.append(f"⚠️  Datasets not ready for training: {not_ready} - run preprocessing first")
        
        # Aggregation strategy recommendations
        if len(self.dataset_metadata) <= 3:
            recommendations.append("💡 Small number of datasets - consider increasing local epochs")
        else:
            recommendations.append("💡 Large number of datasets - consider hierarchical aggregation")
        
        return recommendations
    
    def harmonize_labels(self, strategy: str = "intersection") -> Dict[str, Dict[str, str]]:
        """
        Harmonize labels across datasets using specified strategy.
        
        Args:
            strategy: "intersection" (common labels only), "union" (all labels), or "custom"
            
        Returns:
            Dictionary mapping original labels to harmonized labels for each dataset
        """
        label_analysis = self._analyze_labels()
        harmonization_map = {}
        
        if strategy == "intersection":
            # Use only common labels
            common_labels = label_analysis['common_labels']
            for dataset_name, labels in label_analysis['dataset_labels'].items():
                harmonization_map[dataset_name] = {
                    label_id: label_name for label_id, label_name in labels.items() 
                    if label_id in common_labels
                }
        
        elif strategy == "union":
            # Use all labels, assign unique IDs
            all_label_names = set()
            for labels in label_analysis['dataset_labels'].values():
                all_label_names.update(labels.values())
            
            # Create global label mapping
            global_mapping = {name: str(i) for i, name in enumerate(sorted(all_label_names))}
            
            for dataset_name, labels in label_analysis['dataset_labels'].items():
                harmonization_map[dataset_name] = {
                    label_id: global_mapping[label_name] 
                    for label_id, label_name in labels.items()
                }
        
        self.harmonized_labels = harmonization_map
        logger.info(f"Label harmonization completed using '{strategy}' strategy")
        
        return harmonization_map
    
    def get_client_dataset_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset information for a specific client."""
        for dataset_name, metadata in self.dataset_metadata.items():
            if metadata.get('client_id') == client_id:
                return metadata
        return None
    
    def get_modality_groups(self) -> Dict[str, List[str]]:
        """Group clients by modality for modality-aware aggregation."""
        modality_groups = {}
        
        for dataset_name, metadata in self.dataset_metadata.items():
            client_id = metadata.get('client_id')
            if client_id is None:
                continue
                
            for modality in metadata['modalities']:
                if modality not in modality_groups:
                    modality_groups[modality] = []
                if client_id not in modality_groups[modality]:
                    modality_groups[modality].append(client_id)
        
        return modality_groups
    
    def validate_federation_setup(self) -> Dict[str, Any]:
        """Validate the current federation setup and provide recommendations."""
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check minimum requirements
        if len(self.dataset_metadata) < 2:
            validation_report["valid"] = False
            validation_report["errors"].append("At least 2 datasets required for federation")
        
        # Check dataset readiness
        not_ready = [name for name, meta in self.dataset_metadata.items() if not meta['ready_for_training']]
        if not_ready:
            validation_report["warnings"].append(f"Datasets not ready for training: {not_ready}")
        
        # Check label compatibility
        label_analysis = self._analyze_labels()
        if label_analysis['label_conflicts']:
            validation_report["warnings"].append(f"Label conflicts detected: {list(label_analysis['label_conflicts'].keys())}")
            validation_report["recommendations"].append("Consider running label harmonization")
        
        # Check modality distribution
        modality_analysis = self._analyze_modalities()
        if not modality_analysis['homogeneous_modalities']:
            validation_report["recommendations"].append("Enable modality-aware aggregation for better performance")
        
        return validation_report


def create_multi_dataset_config(client_dataset_mapping: Dict[str, str], 
                               preproc_root: str) -> Dict[str, Any]:
    """
    Create a configuration for multi-dataset federation.
    
    Args:
        client_dataset_mapping: Dictionary mapping client IDs to dataset names
        preproc_root: Root directory containing preprocessed datasets
        
    Returns:
        Configuration dictionary for multi-dataset federation
    """
    manager = DatasetCompatibilityManager()
    
    # Register all datasets
    for client_id, dataset_name in client_dataset_mapping.items():
        dataset_path = os.path.join(preproc_root, dataset_name)
        if os.path.exists(dataset_path):
            manager.register_dataset(dataset_name, dataset_path, client_id)
        else:
            logger.warning(f"Dataset path not found: {dataset_path}")
    
    # Analyze compatibility
    compatibility_report = manager.analyze_compatibility()
    validation_report = manager.validate_federation_setup()
    
    # Create configuration
    config = {
        "client_dataset_mapping": client_dataset_mapping,
        "dataset_metadata": manager.dataset_metadata,
        "compatibility_analysis": compatibility_report,
        "validation_report": validation_report,
        "modality_groups": manager.get_modality_groups(),
        "recommended_settings": {
            "enable_modality_aggregation": not compatibility_report.get("modality_analysis", {}).get("homogeneous_modalities", True),
            "harmonize_labels": len(compatibility_report.get("label_analysis", {}).get("label_conflicts", {})) > 0,
            "aggregation_strategy": "modality_aware" if len(compatibility_report.get("modality_analysis", {}).get("unique_modalities", [])) > 1 else "traditional"
        }
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    manager = DatasetCompatibilityManager()
    
    # Example multi-dataset setup
    preproc_root = "/path/to/nnUNet_preprocessed"
    client_datasets = {
        "0": "Dataset005_Prostate",
        "1": "Dataset009_Spleen", 
        "2": "Dataset027_ACDC"
    }
    
    config = create_multi_dataset_config(client_datasets, preproc_root)
    print("Multi-dataset federation configuration created successfully!")
    print(f"Detected modalities: {config['compatibility_analysis']['modality_analysis']['unique_modalities']}")
    print(f"Recommendations: {config['compatibility_analysis']['recommendations']}")