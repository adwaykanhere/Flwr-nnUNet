# client_app.py

import os
import json
import numpy as np
from typing import Dict
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays

from task import FedNnUNetTrainer
import warnings
warnings.filterwarnings("ignore")


def get_nnunet_preprocessed_path():
    """Get nnUNet preprocessed path from environment variable with fallback."""
    # Try environment variable first
    nnunet_path = os.environ.get("nnUNet_preprocessed")
    if nnunet_path and os.path.exists(nnunet_path):
        print(f"[Client] Using nnUNet preprocessed path from environment: {nnunet_path}")
        return nnunet_path
    
    # Fallback to hardcoded path with warning
    fallback_path = "/local/projects-t3/isaiahlab/nnUNet_preprocessed/"
    if os.path.exists(fallback_path):
        print(f"[Client] WARNING: nnUNet_preprocessed environment variable not set.")
        print(f"[Client] Using fallback path: {fallback_path}")
        print(f"[Client] Please set: export nnUNet_preprocessed=/path/to/your/nnUNet_preprocessed")
        return fallback_path
    
    # Error if no valid path found
    raise RuntimeError(
        f"nnUNet preprocessed directory not found. Please set the nnUNet_preprocessed environment variable:\n"
        f"export nnUNet_preprocessed=/path/to/your/nnUNet_preprocessed\n"
        f"Or ensure the directory exists at: {fallback_path}"
    )

class NnUNet3DFullresClient(NumPyClient):
    """
    Flower NumPyClient that uses FedNnUNetTrainer in 3D fullres mode.
    """

    def __init__(
        self,
        client_id: int,
        plans_path: str,
        dataset_json: str,
        configuration: str,
        dataset_fingerprint: str | None = None,
        max_total_epochs: int = 50,
        local_epochs_per_round: int = 2,
        fold: int = 0,
    ):
        super().__init__()
        self.client_id = client_id
        print(f"[Client {client_id}] Initializing nnUNet trainer with fold {fold}")
        
        # Load JSON files
        import json
        import torch
        with open(plans_path, "r") as f:
            plans_dict = json.load(f)
        with open(dataset_json, "r") as f:
            dataset_dict = json.load(f)
        
        self.trainer = FedNnUNetTrainer(
            plans=plans_dict,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_dict,
            device=torch.device("cuda:0"),
        )
        self.trainer.max_num_epochs = max_total_epochs
        self.local_epochs_per_round = local_epochs_per_round
        self.param_keys = None
        self.dataset_fingerprint_path = dataset_fingerprint or os.path.join(
            os.path.dirname(dataset_json),
            "dataset_fingerprint.json",
        )
        self.dataset_json_path = dataset_json
        self.num_training_cases = self._count_training_cases()
        self.local_fingerprint = self._load_local_fingerprint()
        
        # Best validation tracking for PyTorch model saving
        self.best_validation_dice = 0.0
        self.best_round = 0
        self.output_dir = None

    def _load_local_fingerprint(self) -> dict:
        if self.dataset_fingerprint_path and os.path.exists(self.dataset_fingerprint_path):
            try:
                with open(self.dataset_fingerprint_path, "r") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"[Client {self.client_id}] Could not load fingerprint: {exc}")
        else:
            print(
                f"[Client {self.client_id}] Fingerprint file not found at {self.dataset_fingerprint_path}"
            )
        return {}

    def _extract_modality_info(self) -> dict:
        """Extract modality information from dataset.json and fingerprint.
        Returns only primitive types compatible with Flower's ConfigRecord validation.
        """
        modality_info = {}
        
        try:
            # Load dataset.json to get channel names and modality info
            with open(self.dataset_json_path, "r") as f:
                dataset_dict = json.load(f)
            
            # Extract channel names and flatten to string representation
            channel_names = dataset_dict.get("channel_names", {})
            if channel_names:
                # Convert channel_names dict to string representation for Flower compatibility
                modality_info["channel_names_str"] = json.dumps(channel_names)
                
                # Infer primary modality from channel names
                first_channel_key = list(channel_names.keys())[0] if channel_names else "0"
                first_channel_name = channel_names.get(first_channel_key, "").lower()
                
                if 'ct' in first_channel_name or 'computed' in first_channel_name:
                    modality_info["modality"] = "CT"
                elif 'mr' in first_channel_name or 'magnetic' in first_channel_name or 't1' in first_channel_name or 't2' in first_channel_name:
                    modality_info["modality"] = "MR"
                elif 'pet' in first_channel_name:
                    modality_info["modality"] = "PET"
                elif 'us' in first_channel_name or 'ultrasound' in first_channel_name:
                    modality_info["modality"] = "US"
                else:
                    modality_info["modality"] = "UNKNOWN"
            
            # Extract additional dataset metadata (already primitive types)
            modality_info["dataset_name"] = dataset_dict.get("name", "unknown")
            modality_info["dataset_description"] = dataset_dict.get("description", "")
            modality_info["num_training"] = dataset_dict.get("numTraining", 0)
            modality_info["num_test"] = dataset_dict.get("numTest", 0)
            
            # Add dataset path information for multi-dataset federation
            dataset_path_parts = self.dataset_json_path.split(os.sep)
            for part in reversed(dataset_path_parts):
                if part.startswith("Dataset"):
                    modality_info["dataset_id"] = part
                    break
            else:
                modality_info["dataset_id"] = "unknown"
            
            # Extract modality info from fingerprint if available
            if self.local_fingerprint:
                intensity_props = self.local_fingerprint.get("foreground_intensity_properties_per_channel", {})
                if isinstance(intensity_props, dict) and intensity_props:
                    # Convert list to string for Flower compatibility
                    modality_info["intensity_channels_str"] = json.dumps(list(intensity_props.keys()))
                    
                    # Get intensity statistics for the primary channel and flatten
                    first_channel = list(intensity_props.keys())[0]
                    if first_channel in intensity_props and isinstance(intensity_props[first_channel], dict):
                        channel_stats = intensity_props[first_channel]
                        # Flatten intensity_stats dict to separate primitive fields
                        modality_info["intensity_mean"] = float(channel_stats.get("mean", 0.0))
                        modality_info["intensity_std"] = float(channel_stats.get("std", 0.0))
                        modality_info["intensity_min"] = float(channel_stats.get("min", 0.0))
                        modality_info["intensity_max"] = float(channel_stats.get("max", 0.0))
            
            print(f"[Client {self.client_id}] Extracted modality info: {modality_info}")
            
        except Exception as exc:
            print(f"[Client {self.client_id}] Error extracting modality info: {exc}")
            modality_info = {"modality": "UNKNOWN"}
        
        return modality_info

    def _get_architecture_info(self) -> dict:
        """Extract architecture information for compatibility checking."""
        arch_info = {}
        
        try:
            # Load dataset.json to get input channels and output classes
            with open(self.dataset_json_path, "r") as f:
                dataset_dict = json.load(f)
            
            # Get number of input channels
            channel_names = dataset_dict.get("channel_names", {})
            arch_info["input_channels"] = len(channel_names) if channel_names else 1
            
            # Get number of output classes (labels)
            labels = dataset_dict.get("labels", {})
            # Background is typically label 0, so num_classes = max_label + 1
            if labels:
                label_values = []
                for label_name, label_value in labels.items():
                    if isinstance(label_value, (int, float)):
                        label_values.append(int(label_value))
                    elif isinstance(label_value, list) and len(label_value) > 0:
                        label_values.extend([int(x) for x in label_value if isinstance(x, (int, float))])
                
                arch_info["num_classes"] = max(label_values) + 1 if label_values else 2
            else:
                arch_info["num_classes"] = 2  # Default binary segmentation
            
            # Get patch size from trainer if available
            if hasattr(self.trainer, 'was_initialized') and self.trainer.was_initialized:
                if hasattr(self.trainer, 'configuration_manager'):
                    patch_size = getattr(self.trainer.configuration_manager, 'patch_size', None)
                    if patch_size:
                        arch_info["patch_size"] = list(patch_size)
            
            print(f"[Client {self.client_id}] Architecture info: {arch_info}")
            
        except Exception as exc:
            print(f"[Client {self.client_id}] Error extracting architecture info: {exc}")
            arch_info = {
                "input_channels": 1,
                "num_classes": 2,
                "patch_size": [20, 160, 160]  # Default 3D patch size
            }
        
        return arch_info

    def _filter_backbone_parameters(self, weights_dict: dict) -> dict:
        """
        Filter parameters to exclude first and last layers, keeping only backbone layers.
        For the new aggregation strategy where only middle layers are shared.
        """
        backbone_dict = {}
        excluded_layers = []
        
        for param_name, param_tensor in weights_dict.items():
            exclude_param = False
            exclusion_reason = ""
            
            # Exclude first layer (input layer)
            if self._is_input_layer(param_name):
                exclude_param = True
                exclusion_reason = "first layer (input layer)"
            # Exclude last layer (output layer)
            elif self._is_output_layer(param_name):
                exclude_param = True
                exclusion_reason = "last layer (output layer)"
            
            if exclude_param:
                excluded_layers.append((param_name, exclusion_reason))
                print(f"[Client {self.client_id}] Excluding {param_name}: {exclusion_reason}")
            else:
                backbone_dict[param_name] = param_tensor
        
        print(f"[Client {self.client_id}] Backbone filtering: using {len(backbone_dict)}/{len(weights_dict)} parameters")
        print(f"[Client {self.client_id}] Excluded {len(excluded_layers)} first/last layer parameters")
        
        return backbone_dict
    
    def _is_input_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to an input layer that's sensitive to input channels."""
        # nnUNet input layer patterns - be more specific to avoid false positives
        input_patterns = [
            "_orig_mod.encoder.stages.0.0.convs.0.conv.weight",  # Main nnUNet input layer
            "encoder.stages.0.0.convs.0.conv.weight",  # Without torch.compile wrapper
            "network.encoder.stages.0.0.convs.0.conv.weight",  # Alternative wrapper
        ]
        
        # Check exact matches only - be very strict
        return param_name in input_patterns
    
    def _is_output_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to an output layer that's sensitive to number of classes."""
        # nnUNet output layer patterns - be more specific
        output_patterns = [
            "_orig_mod.decoder.seg_layers",  # Main segmentation layers
            "decoder.seg_layers",  # Without torch.compile wrapper
            "_orig_mod.seg_layers",  # Alternative structure
            "seg_layers",  # Direct segmentation layers
        ]
        
        # Check if parameter name contains any output pattern
        for pattern in output_patterns:
            if pattern in param_name and (".weight" in param_name or ".bias" in param_name):
                return True
        
        return False
    
    def _is_batch_norm_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to a batch normalization layer."""
        bn_patterns = [
            "norm.weight", "norm.bias", "norm.running_mean", "norm.running_var",
            "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var",
            "batch_norm", "batchnorm",
            ".norm1.", ".norm2.",  # Transformer-style normalization
        ]
        
        return any(pattern in param_name for pattern in bn_patterns)

    def _log_parameter_structure(self, weights_dict: dict, context: str):
        """Log detailed parameter structure for debugging."""
        print(f"\n[Client {self.client_id}] ===== PARAMETER STRUCTURE ({context}) =====")
        print(f"[Client {self.client_id}] Total parameters: {len(weights_dict)}")
        
        # Categorize parameters by type
        encoder_params = []
        decoder_params = []
        seg_layer_params = []
        transpconv_params = []
        other_params = []
        
        for param_name, param_tensor in weights_dict.items():
            shape_str = f"{param_tensor.shape}" if hasattr(param_tensor, 'shape') else "unknown"
            
            if "_orig_mod.encoder" in param_name or "encoder" in param_name:
                encoder_params.append((param_name, shape_str))
            elif "_orig_mod.decoder.seg_layers" in param_name or "seg_layers" in param_name:
                seg_layer_params.append((param_name, shape_str))
            elif "_orig_mod.decoder.transpconvs" in param_name or "transpconvs" in param_name:
                transpconv_params.append((param_name, shape_str))
            elif "_orig_mod.decoder" in param_name or "decoder" in param_name:
                decoder_params.append((param_name, shape_str))
            else:
                other_params.append((param_name, shape_str))
        
        print(f"[Client {self.client_id}] Encoder parameters: {len(encoder_params)}")
        if encoder_params:
            print(f"[Client {self.client_id}]   First: {encoder_params[0][0]} -> {encoder_params[0][1]}")
            print(f"[Client {self.client_id}]   Last:  {encoder_params[-1][0]} -> {encoder_params[-1][1]}")
        
        print(f"[Client {self.client_id}] Decoder parameters: {len(decoder_params)}")
        if decoder_params:
            print(f"[Client {self.client_id}]   First: {decoder_params[0][0]} -> {decoder_params[0][1]}")
            print(f"[Client {self.client_id}]   Last:  {decoder_params[-1][0]} -> {decoder_params[-1][1]}")
        
        print(f"[Client {self.client_id}] Transpconv parameters: {len(transpconv_params)}")
        for param_name, shape_str in transpconv_params:
            print(f"[Client {self.client_id}]   {param_name} -> {shape_str}")
        
        print(f"[Client {self.client_id}] Segmentation layer parameters: {len(seg_layer_params)}")
        for param_name, shape_str in seg_layer_params:
            print(f"[Client {self.client_id}]   {param_name} -> {shape_str}")
        
        if other_params:
            print(f"[Client {self.client_id}] Other parameters: {len(other_params)}")
            for param_name, shape_str in other_params[:3]:  # Show first 3
                print(f"[Client {self.client_id}]   {param_name} -> {shape_str}")
        
        print(f"[Client {self.client_id}] =======================================\n")

    def _load_backbone_parameters(self, backbone_parameters: dict):
        """Load backbone parameters while preserving first and last layer weights."""
        try:
            current_weights = self.trainer.get_weights()
            updated_weights = current_weights.copy()
            
            # Update only backbone parameters, keeping first/last layers unchanged
            loaded_count = 0
            for param_name, param_value in backbone_parameters.items():
                if param_name in current_weights:
                    if hasattr(param_value, 'shape') and hasattr(current_weights[param_name], 'shape'):
                        if param_value.shape == current_weights[param_name].shape:
                            updated_weights[param_name] = param_value
                            loaded_count += 1
                        else:
                            print(f"[Client {self.client_id}] Shape mismatch for {param_name}: {param_value.shape} vs {current_weights[param_name].shape}")
                    else:
                        updated_weights[param_name] = param_value
                        loaded_count += 1
            
            self.trainer.set_weights(updated_weights)
            print(f"[Client {self.client_id}] Loaded {loaded_count} backbone parameters")
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error loading backbone parameters: {e}")
            print(f"[Client {self.client_id}] Keeping current parameters as fallback")

    
    
    
    
    
    

    def _count_training_cases(self) -> int:
        """Return number of training cases listed in dataset.json."""
        try:
            with open(self.dataset_json_path, "r") as f:
                data = json.load(f)
            
            # nnUNet dataset.json uses "numTraining" field for total training cases
            num_training = data.get("numTraining", 0)
            print(f"[Client {self.client_id}] Found {num_training} total training cases in dataset.json")
            
            # However, for federated learning, we should return the actual number of cases
            # this client will train on (which depends on the cross-validation split)
            # For now, return the total and we'll get the actual split count from the trainer
            return num_training
        except Exception as exc:
            print(f"[Client {self.client_id}] Could not parse dataset.json: {exc}")
            return 1

    def _get_actual_training_count(self) -> int:
        """Get the actual number of training cases this client is using from the trainer's split."""
        try:
            if hasattr(self.trainer, 'was_initialized') and self.trainer.was_initialized:
                # Get the training split from the trainer
                tr_keys, val_keys = self.trainer.do_split()
                actual_count = len(tr_keys)
                print(f"[Client {self.client_id}] Trainer split: {actual_count} training cases, {len(val_keys)} validation cases")
                return actual_count
            else:
                # Fallback to dataset.json count if trainer not initialized
                print(f"[Client {self.client_id}] Trainer not initialized, falling back to dataset.json count")
                return self.num_training_cases
        except Exception as exc:
            print(f"[Client {self.client_id}] Error getting training count from trainer: {exc}")
            # Fallback to the original dataset count
            return self.num_training_cases if self.num_training_cases > 0 else 25  # Default for prostate dataset

    def _apply_global_fingerprint(self, global_fingerprint: dict):
        """Apply global fingerprint to local dataset_fingerprint.json file."""
        try:
            # Update local fingerprint file with global statistics
            fingerprint_path = self.dataset_fingerprint_path
            if os.path.exists(fingerprint_path):
                with open(fingerprint_path, "w") as f:
                    json.dump(global_fingerprint, f, indent=2)
                print(f"[Client {self.client_id}] Applied global fingerprint to {fingerprint_path}")
            else:
                print(f"[Client {self.client_id}] Warning: fingerprint file not found at {fingerprint_path}")
        except Exception as exc:
            print(f"[Client {self.client_id}] Error applying global fingerprint: {exc}")

    def set_output_directory(self, output_dir: str):
        """Set the output directory for saving PyTorch model checkpoints"""
        self.output_dir = output_dir
        # Create client-specific subdirectory
        import os
        client_output_dir = os.path.join(output_dir, f"client_{self.client_id}")
        
        try:
            os.makedirs(client_output_dir, exist_ok=True)
            self.client_output_dir = client_output_dir
            print(f"[Client {self.client_id}] Output directory set to: {client_output_dir}")
        except PermissionError as e:
            print(f"[Client {self.client_id}] ERROR: Permission denied creating output directory: {client_output_dir}")
            print(f"[Client {self.client_id}] Please either:")
            print(f"[Client {self.client_id}]   1. Set OUTPUT_ROOT environment variable to a writable directory:")
            print(f"[Client {self.client_id}]      export OUTPUT_ROOT=\"./federated_models\"")
            print(f"[Client {self.client_id}]   2. Or create the directory with proper permissions:")
            print(f"[Client {self.client_id}]      sudo mkdir -p {output_dir} && sudo chown $USER {output_dir}")
            print(f"[Client {self.client_id}] Model saving will be disabled.")
            self.client_output_dir = None
        except Exception as e:
            print(f"[Client {self.client_id}] ERROR: Failed to create output directory: {e}")
            print(f"[Client {self.client_id}] Model saving will be disabled.")
            self.client_output_dir = None

    def get_parameters(self, config) -> NDArrays:
        """
        Return current model parameters as a list of numpy arrays.
        For new strategy: only return backbone layers (exclude first/last layers).
        """
        if not self.trainer.was_initialized:
            self.trainer.initialize()

        weights_dict = self.trainer.get_weights()
        
        # Log detailed parameter structure for debugging
        self._log_parameter_structure(weights_dict, "get_parameters")
        
        # Filter to backbone parameters only (exclude first and last layers)
        backbone_weights_dict = self._filter_backbone_parameters(weights_dict)
        
        self.param_keys = list(backbone_weights_dict.keys())
        print(f"[Client {self.client_id}] Sending {len(self.param_keys)} backbone parameters (excluded first/last layers)")
        return list(backbone_weights_dict.values())

    def _warmup_first_last_layers(self, warmup_epochs: int):
        """
        Warm up first and last layers by training them locally for specified epochs.
        """
        print(f"[Client {self.client_id}] Starting warmup: training first/last layers for {warmup_epochs} epochs")
        
        # Train locally without sharing weights
        for epoch in range(warmup_epochs):
            print(f"[Client {self.client_id}] Warmup epoch {epoch + 1}/{warmup_epochs}")
            self.trainer.run_training_round(1)
        
        self.is_warmed_up = True
        print(f"[Client {self.client_id}] Warmup complete - first/last layers trained for {warmup_epochs} epochs")

    def fit(self, parameters: NDArrays, config):
        """
        Receive global backbone params, do local training, return updated params + metrics.
        Implements new backbone aggregation strategy with warmup logic.
        """
        federated_round = config.get("server_round", 1)
        
        # Handle preprocessing round (federated_round = -2) - share fingerprint only
        if federated_round == -2:
            print(f"[Client {self.client_id}] Preprocessing round - sharing fingerprint")
            if not self.trainer.was_initialized:
                self.trainer.initialize()
            
            # Get initial backbone parameters for consistency
            if self.param_keys is None:
                local_sd = self.trainer.get_weights()
                backbone_sd = self._filter_backbone_parameters(local_sd)
                self.param_keys = list(backbone_sd.keys())
                
            initial_params = [backbone_sd[k] for k in self.param_keys]
            
            # Create a simple fingerprint summary for metrics
            fp_summary = {}
            if self.local_fingerprint:
                fp_summary["num_cases"] = len(self.local_fingerprint.get("spacings", []))
                # Get first modality stats if available
                if "foreground_intensity_properties_per_channel" in self.local_fingerprint:
                    intensity_props = self.local_fingerprint["foreground_intensity_properties_per_channel"]
                    if isinstance(intensity_props, dict) and intensity_props:
                        first_mod = list(intensity_props.keys())[0]
                        if first_mod in intensity_props and isinstance(intensity_props[first_mod], dict):
                            fp_summary["mean_intensity"] = float(intensity_props[first_mod].get("mean", 0.0))
            
            # Get actual training count for preprocessing phase too
            actual_training_cases = self._get_actual_training_count()
            
            # Extract modality information
            modality_info = self._extract_modality_info()
            
            # Extract architecture information
            arch_info = self._get_architecture_info()
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "preprocessing_complete": True,
                "fingerprint_cases": fp_summary.get("num_cases", 0),
                "fingerprint_mean": fp_summary.get("mean_intensity", 0.0),
                "actual_training_cases": actual_training_cases,
                "is_warmup": False
            }
            # Add modality and architecture information to metrics
            metrics.update(modality_info)
            metrics.update(arch_info)
            return initial_params, actual_training_cases, metrics
        
        # Handle initialization round (federated_round = -1) - apply global backbone parameters
        if federated_round == -1:
            print(f"[Client {self.client_id}] Initialization round")
            
            if not self.trainer.was_initialized:
                self.trainer.initialize()
                
            if self.param_keys is None:
                local_sd = self.trainer.get_weights()
                backbone_sd = self._filter_backbone_parameters(local_sd)
                self.param_keys = list(backbone_sd.keys())
                
            # Apply received global backbone parameters
            if parameters:
                backbone_params = {}
                for k, arr in zip(self.param_keys, parameters):
                    backbone_params[k] = arr
                
                # Load backbone parameters while preserving first/last layers
                self._load_backbone_parameters(backbone_params)
                
            updated_dict = self.trainer.get_weights()
            backbone_dict = self._filter_backbone_parameters(updated_dict)
            updated_params = [backbone_dict[k] for k in self.param_keys]
            
            # Get actual training count for initialization phase too
            actual_training_cases = self._get_actual_training_count()
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "initialization_complete": True,
                "actual_training_cases": actual_training_cases,
                "is_warmup": False
            }
            return updated_params, actual_training_cases, metrics

        # Regular training rounds (federated_round >= 0)
        print(f"[Client {self.client_id}] Training round {federated_round}")
        
        # Check if this is round 0 (warmup round)
        is_warmup_round = (federated_round == 0)
        
        if self.param_keys is None:
            local_sd = self.trainer.get_weights()
            backbone_sd = self._filter_backbone_parameters(local_sd)
            self.param_keys = list(backbone_sd.keys())

        # Handle warmup logic for round 0
        if is_warmup_round and not self.is_warmed_up:
            print(f"[Client {self.client_id}] Round 0: Starting warmup phase")
            # Warm up first and last layers locally
            self._warmup_first_last_layers(self.warmup_epochs)
            
            # No parameter loading from server in warmup round
            # Just return current backbone parameters after warmup
            updated_dict = self.trainer.get_weights()
            backbone_dict = self._filter_backbone_parameters(updated_dict)
            updated_params = [backbone_dict[k] for k in self.param_keys]
            
            # Get actual number of training cases
            actual_training_cases = self._get_actual_training_count()
            
            # Training metrics for warmup round
            final_loss = (
                self.trainer.all_train_losses[-1] if self.trainer.all_train_losses else 0.0
            )

            metrics = {
                "client_id": self.client_id,
                "loss": final_loss,
                "federated_round": federated_round,
                "local_epochs_completed": self.warmup_epochs,
                "actual_training_cases": actual_training_cases,
                "is_warmup": True,
                "warmup_complete": True
            }
            
            return updated_params, actual_training_cases, metrics
        
        # Regular training rounds (federated_round > 0)
        # Load backbone parameters from server
        if parameters:
            backbone_params = {}
            for k, arr in zip(self.param_keys, parameters):
                backbone_params[k] = arr
            
            # Load backbone parameters while preserving first/last layers
            self._load_backbone_parameters(backbone_params)

        # Local training
        local_epochs = config.get("local_epochs", 1)
        self.trainer.run_training_round(local_epochs)

        # Return updated backbone parameters only
        updated_dict = self.trainer.get_weights()
        backbone_dict = self._filter_backbone_parameters(updated_dict)
        updated_params = [backbone_dict[k] for k in self.param_keys]

        # Get actual number of training cases from the trainer's split
        actual_training_cases = self._get_actual_training_count()
        print(f"[Client {self.client_id}] Using {actual_training_cases} training cases for backbone aggregation")

        # Training metrics
        final_loss = (
            self.trainer.all_train_losses[-1] if self.trainer.all_train_losses else 0.0
        )

        metrics = {
            "client_id": self.client_id,
            "loss": final_loss,
            "federated_round": federated_round,
            "local_epochs_completed": local_epochs,
            "actual_training_cases": actual_training_cases,
            "is_warmup": False
        }
        
        # Add modality information to training metrics if requested
        if config.get("enable_modality_metadata", False):
            modality_info = self._extract_modality_info()
            metrics.update(modality_info)
        
        # Run validation if requested
        should_validate = config.get("validate", False)
        if should_validate:
            try:
                print(f"[Client {self.client_id}] Running validation...")
                validation_results = self.trainer.run_validation_round()
                
                # Extract only primitive values for Flower compatibility
                current_dice = validation_results.get('mean', 0.0)
                metrics["validation_dice_mean"] = float(current_dice)
                metrics["validation_num_batches"] = validation_results.get('num_batches', 0)
                
                # Flatten per_label scores to individual primitive fields
                per_label_scores = validation_results.get('per_label', {})
                for label, score in per_label_scores.items():
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        metrics[f"validation_dice_label_{label}"] = float(score)
                
                print(f"[Client {self.client_id}] Validation Dice: {current_dice:.4f}")
                
                # Save PyTorch model if validation improved and output directory is set
                if hasattr(self, 'client_output_dir') and self.client_output_dir:
                    is_best = current_dice > self.best_validation_dice
                    if is_best:
                        self.best_validation_dice = current_dice
                        self.best_round = federated_round
                        print(f"[Client {self.client_id}] New best validation Dice: {current_dice:.4f}")
                    
                    # Save PyTorch checkpoint only if it's the best model
                    if is_best:
                        try:
                            checkpoint_path = self.trainer.save_best_checkpoint_pytorch(
                                output_dir=self.client_output_dir,
                                round_num=federated_round,
                                validation_dice=current_dice,
                                is_best=True
                            )
                            if checkpoint_path:
                                metrics["pytorch_checkpoint_saved"] = checkpoint_path
                                metrics["best_model_updated"] = True
                                print(f"[Client {self.client_id}] Saved best model checkpoint: {checkpoint_path}")
                        except Exception as save_e:
                            print(f"[Client {self.client_id}] Failed to save PyTorch checkpoint: {save_e}")
                    else:
                        print(f"[Client {self.client_id}] Validation Dice ({current_dice:.4f}) <= best ({self.best_validation_dice:.4f}), not saving model")
                        
            except Exception as e:
                print(f"[Client {self.client_id}] Validation failed: {e}")
                # Continue without validation metrics
        
        return updated_params, actual_training_cases, metrics

    def evaluate(self, parameters: NDArrays, config):
        """
        Evaluate global model on local validation set (optional).
        """
        if self.param_keys is None:
            local_sd = self.trainer.get_weights()
            self.param_keys = list(local_sd.keys())

        # Convert list->dict, set local model
        new_sd = {}
        for k, arr in zip(self.param_keys, parameters):
            new_sd[k] = arr
        self.trainer.set_weights(new_sd)

        # Example local validation
        val_loss = 0.5  # or run actual inference
        return val_loss, self.num_training_cases, {"val_loss": val_loss}


def get_client_dataset_config(client_id: int, context: Context) -> Dict[str, str]:
    """
    Get dataset configuration for a specific client.
    Supports both single-dataset and multi-dataset federation setups.
    Now supports dataset-path and dataset-name from node-config with highest priority.
    """
    # Priority 1: Check for dataset-path in node-config (full path)
    dataset_path = context.node_config.get("dataset-path") if hasattr(context, 'node_config') and context.node_config else None
    if dataset_path:
        # Extract dataset name from full path
        import os
        dataset_name = os.path.basename(dataset_path.rstrip('/'))
        if dataset_name.startswith('Dataset'):
            print(f"[Client {client_id}] Using dataset path from node-config: {dataset_path}")
            return {"dataset_name": dataset_name, "source": "node_config_path", "dataset_path": dataset_path}
        else:
            print(f"[Client {client_id}] Warning: Invalid dataset path format in node-config: {dataset_path}")
    
    # Priority 2: Check for dataset-name in node-config
    dataset_name = context.node_config.get("dataset-name") if hasattr(context, 'node_config') and context.node_config else None
    if dataset_name:
        print(f"[Client {client_id}] Using dataset name from node-config: {dataset_name}")
        return {"dataset_name": dataset_name, "source": "node_config_name"}
    
    # Priority 3: Check for multi-dataset configuration from environment
    import os
    client_datasets_json = os.environ.get("CLIENT_DATASETS")
    if client_datasets_json:
        try:
            import json
            client_datasets = json.loads(client_datasets_json)
            print(f"[Client {client_id}] Parsed CLIENT_DATASETS: {client_datasets}")
            
            # Ensure client_datasets is a dictionary
            if not isinstance(client_datasets, dict):
                print(f"[Client {client_id}] Warning: CLIENT_DATASETS is not a dictionary: {type(client_datasets)}")
                print(f"[Client {client_id}] Raw value: {client_datasets}")
            else:
                client_key = str(client_id)
                if client_key in client_datasets:
                    dataset_name = client_datasets[client_key]
                    print(f"[Client {client_id}] Using multi-dataset config: {dataset_name}")
                    return {"dataset_name": dataset_name, "source": "multi_dataset"}
                else:
                    print(f"[Client {client_id}] Client key '{client_key}' not found in CLIENT_DATASETS")
                    print(f"[Client {client_id}] Available keys: {list(client_datasets.keys())}")
        except json.JSONDecodeError as e:
            print(f"[Client {client_id}] Warning: Invalid CLIENT_DATASETS JSON: {e}")
        except Exception as e:
            print(f"[Client {client_id}] Error parsing CLIENT_DATASETS: {e}")
    
    # Check for client-specific environment variable
    client_task_env = f"CLIENT_{client_id}_DATASET"
    client_dataset = os.environ.get(client_task_env)
    if client_dataset:
        print(f"[Client {client_id}] Using client-specific dataset: {client_dataset}")
        return {"dataset_name": client_dataset, "source": "client_specific"}
    
    # Fallback to global TASK_NAME
    task_name = os.environ.get("TASK_NAME", "Dataset005_Prostate")
    print(f"[Client {client_id}] Using global dataset: {task_name}")
    return {"dataset_name": task_name, "source": "global"}

def client_fn(context: Context):
    """
    This callback is used by Flower 1.13 to create the client instance.
    Supports multi-dataset federation with client-specific dataset assignment.
    """
    # Handle node_config whether it's a string or dict
    try:
        if isinstance(context.node_config, dict):
            # Standard dictionary access
            client_id = context.node_config.get("partition-id", 0)
        elif isinstance(context.node_config, str):
            # Parse string format: "partition-id=0"
            client_id = 0  # default
            for pair in context.node_config.split():
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    if key.strip() == "partition-id":
                        client_id = int(value.strip())
                        break
        else:
            print(f"[Client] Warning: Unexpected node_config type: {type(context.node_config)}")
            print(f"[Client] node_config value: {context.node_config}")
            client_id = 0
    except Exception as e:
        print(f"[Client] Error parsing node_config: {e}")
        print(f"[Client] node_config type: {type(context.node_config)}")
        print(f"[Client] node_config value: {context.node_config}")
        client_id = 0
    
    # Get dataset configuration for this client
    dataset_config = get_client_dataset_config(client_id, context)
    task_name = dataset_config["dataset_name"]
    
    print(f"[Client {client_id}] Initializing with dataset: {task_name}")
    print(f"[Client {client_id}] Dataset source: {dataset_config['source']}")

    # Determine dataset path - use full path if provided, otherwise construct from preprocessed root
    if "dataset_path" in dataset_config:
        dataset_full_path = dataset_config["dataset_path"]
        print(f"[Client {client_id}] Using explicit dataset path: {dataset_full_path}")
    else:
        preproc_root = get_nnunet_preprocessed_path()
        dataset_full_path = os.path.join(preproc_root, task_name)
        print(f"[Client {client_id}] Constructed dataset path: {dataset_full_path}")
    
    plans_path = os.path.join(dataset_full_path, "nnUNetPlans.json")
    dataset_json = os.path.join(dataset_full_path, "dataset.json")
    dataset_fp = os.path.join(dataset_full_path, "dataset_fingerprint.json")
    
    # Validate dataset paths
    required_files = [plans_path, dataset_json]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        error_msg = f"[Client {client_id}] Missing required dataset files for {task_name}: {missing_files}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Extract fold parameter from node-config (default to 0 for backward compatibility)
    fold = context.node_config.get("fold", 0) if hasattr(context, 'node_config') and context.node_config else 0
    try:
        fold = int(fold)  # Ensure fold is an integer
        print(f"[Client {client_id}] Using fold: {fold}")
    except (ValueError, TypeError):
        print(f"[Client {client_id}] Warning: Invalid fold value '{fold}', using default fold=0")
        fold = 0
    
    configuration = os.environ.get("NNUNET_CONFIG", "3d_fullres")
    out_root = os.environ.get("OUTPUT_ROOT", "./federated_models")
    output_folder = os.path.join(out_root, f"client_{client_id}")

    # Create the client
    client = NnUNet3DFullresClient(
        client_id=client_id,
        plans_path=plans_path,
        dataset_json=dataset_json,
        configuration=configuration,
        dataset_fingerprint=dataset_fp,
        max_total_epochs=50,
        local_epochs_per_round=2,
        fold=fold,
    )
    
    # Set output directory for model saving
    client.set_output_directory(out_root)
    print(f"[Client {client_id}] Model saving enabled to: {out_root}")
    
    return client.to_client()


# Flower 1.13+ recommended usage: a ClientApp
app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    print("This is a Flower ClientApp. Typically run with:")
    print("flower-supernode --client-app=client_app.py:app")
