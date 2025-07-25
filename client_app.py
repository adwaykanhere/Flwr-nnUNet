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

    def _filter_incompatible_parameters(self, weights_dict: dict, arch_info: dict, config: dict) -> dict:
        """
        Filter out parameters that may cause architecture mismatches.
        Based on FedBN approach - exclude layers likely to have different architectures.
        
        Common incompatible layer patterns in nnUNet:
        - Input layers: sensitive to number of input channels
        - Output layers: sensitive to number of output classes  
        - Normalization layers: may have different statistics
        """
        filtered_dict = {}
        excluded_layers = []
        
        # Get expected architecture constraints from config if available
        expected_input_channels = config.get("expected_input_channels", arch_info.get("input_channels"))
        expected_num_classes = config.get("expected_num_classes", arch_info.get("num_classes"))
        
        print(f"[Client {self.client_id}] Filtering parameters with arch constraints:")
        print(f"  Client input_channels: {arch_info.get('input_channels')}")
        print(f"  Client num_classes: {arch_info.get('num_classes')}")
        print(f"  Expected input_channels: {expected_input_channels}")
        print(f"  Expected num_classes: {expected_num_classes}")
        
        for param_name, param_tensor in weights_dict.items():
            exclude_param = False
            exclusion_reason = ""
            
            # Check for input layer incompatibility
            # nnUNet typically has input layers with patterns like "encoder.stages.0.0.convs.0.conv.weight"
            if self._is_input_layer(param_name) and expected_input_channels is not None:
                if arch_info.get("input_channels") != expected_input_channels:
                    exclude_param = True
                    exclusion_reason = f"input channel mismatch: {arch_info.get('input_channels')} vs {expected_input_channels}"
            
            # Check for output layer incompatibility
            # nnUNet output layers typically have patterns like "decoder.stages.*.conv_out.weight" or "seg_layers.*.weight"
            elif self._is_output_layer(param_name) and expected_num_classes is not None:
                if arch_info.get("num_classes") != expected_num_classes:
                    exclude_param = True
                    exclusion_reason = f"output class mismatch: {arch_info.get('num_classes')} vs {expected_num_classes}"
            
            # Check for batch normalization layers (FedBN approach excludes these)
            elif self._is_batch_norm_layer(param_name):
                # For now, include BN layers but could exclude for more aggressive filtering
                # exclude_param = True
                # exclusion_reason = "batch normalization layer (dataset-specific statistics)"
                pass
            
            if exclude_param:
                excluded_layers.append((param_name, exclusion_reason))
                print(f"[Client {self.client_id}] Excluding {param_name}: {exclusion_reason}")
            else:
                filtered_dict[param_name] = param_tensor
        
        if excluded_layers:
            print(f"[Client {self.client_id}] Excluded {len(excluded_layers)} incompatible parameters")
            for param_name, reason in excluded_layers[:5]:  # Show first 5
                print(f"  - {param_name}: {reason}")
            if len(excluded_layers) > 5:
                print(f"  ... and {len(excluded_layers) - 5} more")
        else:
            print(f"[Client {self.client_id}] No incompatible parameters detected")
        
        return filtered_dict
    
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

    def _check_direct_compatibility(self, parameters_dict: dict) -> bool:
        """Check if parameters can be loaded directly without adaptation."""
        try:
            current_weights = self.trainer.get_weights()
            
            # Check if all parameters have compatible shapes
            for param_name, param_value in parameters_dict.items():
                if param_name not in current_weights:
                    print(f"[Client {self.client_id}] Parameter {param_name} not found in current model")
                    return False
                
                current_param = current_weights[param_name]
                
                if not hasattr(param_value, 'shape') or not hasattr(current_param, 'shape'):
                    continue
                
                if param_value.shape != current_param.shape:
                    print(f"[Client {self.client_id}] Shape incompatibility: {param_name} {param_value.shape} vs {current_param.shape}")
                    return False
            
            print(f"[Client {self.client_id}] All parameters are directly compatible")
            return True
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error checking compatibility: {e}")
            return False

    def _adapt_fedma_parameters(self, fedma_weights: dict, config: dict) -> dict:
        """
        Adapt FedMA harmonized parameters to client-specific architecture.
        Handles channel count differences and class count differences.
        """
        current_weights = self.trainer.get_weights()
        adapted_weights = {}
        
        print(f"[Client {self.client_id}] Adapting FedMA parameters to local architecture")
        print(f"[Client {self.client_id}] FedMA provides {len(fedma_weights)} parameters")
        print(f"[Client {self.client_id}] Local model expects {len(current_weights)} parameters")
        
        # Log parameter structure comparison
        self._log_parameter_structure(fedma_weights, "FedMA_input")
        self._log_parameter_structure(current_weights, "Local_model")
        
        # Get client architecture info
        arch_info = self._get_architecture_info()
        client_input_channels = arch_info.get('input_channels', 1)
        client_num_classes = arch_info.get('num_classes', 2)
        
        for param_name, fedma_param in fedma_weights.items():
            if param_name not in current_weights:
                print(f"[Client {self.client_id}] Skipping unknown parameter: {param_name}")
                continue
            
            current_param = current_weights[param_name]
            
            # Handle different parameter types
            if self._is_input_layer(param_name):
                adapted_param = self._adapt_input_layer(
                    fedma_param, current_param, client_input_channels
                )
            elif self._is_output_layer(param_name):
                adapted_param = self._adapt_output_layer(
                    fedma_param, current_param, client_num_classes
                )
            else:
                # Middle layers - use as-is if compatible, otherwise adapt
                adapted_param = self._adapt_middle_layer(fedma_param, current_param)
            
            # Validate adapted parameter before adding
            if self._validate_adapted_parameter(adapted_param, current_param, param_name):
                adapted_weights[param_name] = adapted_param
            else:
                print(f"[Client {self.client_id}] Validation failed for {param_name}, using current parameter")
                adapted_weights[param_name] = current_param
        
        # Check for missing parameters in the adapted weights
        missing_params = []
        for param_name in current_weights.keys():
            if param_name not in adapted_weights:
                missing_params.append(param_name)
        
        if missing_params:
            print(f"[Client {self.client_id}] WARNING: {len(missing_params)} parameters will be missing!")
            print(f"[Client {self.client_id}] Adding missing parameters from current model to prevent errors...")
            
            for param_name in missing_params[:10]:  # Show first 10
                print(f"[Client {self.client_id}]   Missing: {param_name}")
            if len(missing_params) > 10:
                print(f"[Client {self.client_id}]   ... and {len(missing_params) - 10} more")
            
            # Add missing parameters from current model to prevent loading errors
            for param_name in missing_params:
                adapted_weights[param_name] = current_weights[param_name]
            
            print(f"[Client {self.client_id}] Added {len(missing_params)} missing parameters from current model")
        
        print(f"[Client {self.client_id}] Adapted {len(adapted_weights)} FedMA parameters")
        print(f"[Client {self.client_id}] Expected {len(current_weights)} parameters")
        return adapted_weights
    
    def _should_attempt_fedma_adaptation(self, parameters_dict: dict) -> bool:
        """Determine if FedMA adaptation should be attempted or if we should use safer fallback."""
        try:
            current_weights = self.trainer.get_weights()
            
            # Count major structural differences
            major_differences = 0
            
            # Check if number of parameters is vastly different
            param_count_ratio = len(parameters_dict) / len(current_weights)
            if param_count_ratio < 0.5 or param_count_ratio > 2.0:
                major_differences += 1
                print(f"[Client {self.client_id}] Major parameter count difference: {len(parameters_dict)} vs {len(current_weights)}")
            
            # Check for major architectural differences (decoder stages, transpconvs, seg_layers)
            current_decoder_stages = len([p for p in current_weights.keys() if "decoder.stages" in p])
            received_decoder_stages = len([p for p in parameters_dict.keys() if "decoder.stages" in p])
            
            current_seg_layers = len([p for p in current_weights.keys() if "seg_layers" in p])
            received_seg_layers = len([p for p in parameters_dict.keys() if "seg_layers" in p])
            
            if abs(current_decoder_stages - received_decoder_stages) > 20:  # Threshold for major difference
                major_differences += 1
                print(f"[Client {self.client_id}] Major decoder stage difference: {received_decoder_stages} vs {current_decoder_stages}")
            
            if abs(current_seg_layers - received_seg_layers) > 5:  # Threshold for major difference
                major_differences += 1
                print(f"[Client {self.client_id}] Major seg_layers difference: {received_seg_layers} vs {current_seg_layers}")
            
            # Only attempt FedMA if differences are minor
            should_attempt = major_differences == 0
            
            if not should_attempt:
                print(f"[Client {self.client_id}] Too many major differences ({major_differences}), using safe fallback")
            
            return should_attempt
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error assessing FedMA suitability: {e}")
            return False
    
    def _apply_safe_parameter_loading(self, parameters_dict: dict):
        """Apply safe parameter loading using only compatible parameters."""
        try:
            current_weights = self.trainer.get_weights()
            compatible_weights = {}
            
            # Only use parameters that have exact shape matches
            compatible_count = 0
            for param_name, param_value in parameters_dict.items():
                if param_name in current_weights:
                    current_param = current_weights[param_name]
                    if (hasattr(param_value, 'shape') and hasattr(current_param, 'shape') and 
                        param_value.shape == current_param.shape):
                        compatible_weights[param_name] = param_value
                        compatible_count += 1
            
            print(f"[Client {self.client_id}] Safe loading: using {compatible_count}/{len(parameters_dict)} compatible parameters")
            
            # Load compatible parameters, keep current weights for incompatible ones
            if compatible_weights:
                # Create full weight dict with current weights as base
                full_weights = current_weights.copy()
                full_weights.update(compatible_weights)
                self.trainer.set_weights(full_weights)
                print(f"[Client {self.client_id}] Successfully loaded {len(compatible_weights)} compatible parameters")
            else:
                print(f"[Client {self.client_id}] No compatible parameters found, keeping current weights")
                
        except Exception as e:
            print(f"[Client {self.client_id}] Error in safe parameter loading: {e}")
            print(f"[Client {self.client_id}] Keeping current parameters as fallback")
    
    def _validate_adapted_parameter(self, adapted_param, current_param, param_name):
        """Validate that adapted parameter is compatible with current parameter."""
        try:
            # Check basic shape compatibility
            if not hasattr(adapted_param, 'shape') or not hasattr(current_param, 'shape'):
                return False
            
            if adapted_param.shape != current_param.shape:
                print(f"[Client {self.client_id}] Shape mismatch for {param_name}: {adapted_param.shape} vs {current_param.shape}")
                return False
            
            # Check for NaN or infinite values
            import numpy as np
            if np.any(np.isnan(adapted_param)) or np.any(np.isinf(adapted_param)):
                print(f"[Client {self.client_id}] Invalid values (NaN/Inf) in adapted parameter {param_name}")
                return False
            
            return True
            
        except Exception as e:
            print(f"[Client {self.client_id}] Validation error for {param_name}: {e}")
            return False
    
    def _adapt_input_layer(self, fedma_param, current_param, client_channels):
        """Adapt harmonized input layer to client's channel count."""
        if fedma_param.shape == current_param.shape:
            return fedma_param
        
        fedma_channels = fedma_param.shape[1]
        
        if client_channels <= fedma_channels:
            # Take subset of channels
            adapted_param = fedma_param[:, :client_channels]
            print(f"[Client {self.client_id}] Input layer: reduced {fedma_channels} → {client_channels} channels")
        else:
            # Need to expand channels - repeat last channel
            import numpy as np
            extra_channels = client_channels - fedma_channels
            last_channel = fedma_param[:, -1:].repeat(extra_channels, axis=1)
            adapted_param = np.concatenate([fedma_param, last_channel], axis=1)
            print(f"[Client {self.client_id}] Input layer: expanded {fedma_channels} → {client_channels} channels")
        
        return adapted_param
    
    def _adapt_output_layer(self, fedma_param, current_param, client_classes):
        """Adapt harmonized output layer to client's class count."""
        if fedma_param.shape == current_param.shape:
            return fedma_param
        
        fedma_classes = fedma_param.shape[0]
        
        if client_classes <= fedma_classes:
            # Take subset of classes
            adapted_param = fedma_param[:client_classes]
            print(f"[Client {self.client_id}] Output layer: reduced {fedma_classes} → {client_classes} classes")
        else:
            # Need to expand classes - initialize new classes with small random values
            import numpy as np
            extra_classes = client_classes - fedma_classes
            new_class_shape = (extra_classes,) + fedma_param.shape[1:]
            new_classes = np.random.normal(0, 0.01, new_class_shape).astype(fedma_param.dtype)
            adapted_param = np.concatenate([fedma_param, new_classes], axis=0)
            print(f"[Client {self.client_id}] Output layer: expanded {fedma_classes} → {client_classes} classes")
        
        return adapted_param
    
    def _adapt_middle_layer(self, fedma_param, current_param):
        """Adapt middle layer parameters."""
        if fedma_param.shape == current_param.shape:
            return fedma_param
        
        # For middle layers with shape mismatches, try adaptive approaches
        import numpy as np
        
        if fedma_param.ndim == current_param.ndim:
            # Same number of dimensions - try to adapt each dimension
            adapted_shape = current_param.shape
            
            # Simple approach: crop or pad to match current shape
            slices = []
            for fedma_dim, current_dim in zip(fedma_param.shape, adapted_shape):
                if fedma_dim >= current_dim:
                    # Crop
                    slices.append(slice(0, current_dim))
                else:
                    # Will need padding later
                    slices.append(slice(None))
            
            # Apply cropping
            adapted_param = fedma_param[tuple(slices)]
            
            # Apply padding if needed
            if adapted_param.shape != adapted_shape:
                padding = []
                for adapted_dim, current_dim in zip(adapted_param.shape, adapted_shape):
                    pad_amount = current_dim - adapted_dim
                    padding.append((0, max(0, pad_amount)))
                
                adapted_param = np.pad(adapted_param, padding, mode='constant', constant_values=0)
            
            print(f"[Client {self.client_id}] Middle layer: adapted {fedma_param.shape} → {adapted_shape}")
            return adapted_param
        else:
            # Shape incompatible - use current parameters
            print(f"[Client {self.client_id}] Middle layer: incompatible shapes {fedma_param.shape} vs {current_param.shape}, keeping current")
            return current_param

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
        Return current model parameters as a list of numpy arrays
        (ordered consistently for Flower).
        Implements FedBN-style selective parameter exclusion for architecture compatibility.
        """
        if not self.trainer.was_initialized:
            self.trainer.initialize()

        weights_dict = self.trainer.get_weights()
        
        # Log detailed parameter structure for debugging
        self._log_parameter_structure(weights_dict, "get_parameters")
        
        # Check if FedMA harmonization is available
        fedma_enabled = config.get("fedma_enabled", True)
        
        if fedma_enabled:
            # Use full parameter set for FedMA (it handles harmonization)
            self.param_keys = list(weights_dict.keys())
            print(f"[Client {self.client_id}] Using {len(self.param_keys)} parameters for FedMA harmonization")
            return list(weights_dict.values())
        else:
            # Fallback to architecture-aware parameter exclusion
            exclude_incompatible = config.get("exclude_incompatible_layers", True)
            
            if exclude_incompatible:
                # Get architecture info for compatibility filtering
                arch_info = self._get_architecture_info()
                
                # Filter out parameters that may cause architecture mismatches
                filtered_weights_dict = self._filter_incompatible_parameters(
                    weights_dict, arch_info, config
                )
                
                self.param_keys = list(filtered_weights_dict.keys())
                print(f"[Client {self.client_id}] Using {len(self.param_keys)} compatible parameters (excluded {len(weights_dict) - len(filtered_weights_dict)} incompatible)")
                return list(filtered_weights_dict.values())
            else:
                self.param_keys = list(weights_dict.keys())
                return list(weights_dict.values())

    def fit(self, parameters: NDArrays, config):
        """
        Receive global model params, do local partial training, return updated params + metrics.
        Implements kaapana-style federated training with fingerprint handling.
        """
        federated_round = config.get("server_round", 1)
        
        # Handle preprocessing round (federated_round = -2) - share fingerprint only
        if federated_round == -2:
            print(f"[Client {self.client_id}] Preprocessing round - sharing fingerprint")
            if not self.trainer.was_initialized:
                self.trainer.initialize()
            
            # Get initial parameters for consistency
            if self.param_keys is None:
                local_sd = self.trainer.get_weights()
                self.param_keys = list(local_sd.keys())
                
            initial_params = [local_sd[k] for k in self.param_keys]
            
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
                "actual_training_cases": actual_training_cases
            }
            # Add modality and architecture information to metrics
            metrics.update(modality_info)
            metrics.update(arch_info)
            return initial_params, actual_training_cases, metrics
        
        # Handle initialization round (federated_round = -1) - apply global fingerprint
        if federated_round == -1:
            print(f"[Client {self.client_id}] Initialization round")
            # global_fingerprint = config.get("global_fingerprint", {})
            # if global_fingerprint:
            #     self._apply_global_fingerprint(global_fingerprint)
            
            if not self.trainer.was_initialized:
                self.trainer.initialize()
                
            if self.param_keys is None:
                local_sd = self.trainer.get_weights()
                self.param_keys = list(local_sd.keys())
                
            # Apply received global parameters
            if parameters:
                new_sd = {}
                for k, arr in zip(self.param_keys, parameters):
                    new_sd[k] = arr
                
                # Apply FedMA-aware parameter loading for initialization too
                fedma_enabled = config.get("fedma_enabled", True)
                
                if fedma_enabled:
                    # First try to load parameters directly to see if they're compatible
                    compatible_direct = self._check_direct_compatibility(new_sd)
                    
                    if compatible_direct:
                        print(f"[Client {self.client_id}] Initialization parameters are directly compatible, skipping FedMA adaptation")
                        self.trainer.set_weights(new_sd)
                    else:
                        print(f"[Client {self.client_id}] Initialization parameters incompatible")
                        # Check if we should attempt FedMA adaptation or use safer fallback
                        if self._should_attempt_fedma_adaptation(new_sd):
                            print(f"[Client {self.client_id}] Attempting FedMA adaptation for initialization")
                            adapted_weights = self._adapt_fedma_parameters(new_sd, config)
                            self.trainer.set_weights(adapted_weights)
                        else:
                            print(f"[Client {self.client_id}] Using safe parameter filtering for initialization")
                            self._apply_safe_parameter_loading(new_sd)
                else:
                    # Traditional compatible parameter loading
                    current_weights = self.trainer.get_weights()
                    compatible_weights = {}
                    
                    for param_name, param_value in new_sd.items():
                        if param_name in current_weights:
                            # Check if shapes are compatible
                            if hasattr(param_value, 'shape') and hasattr(current_weights[param_name], 'shape'):
                                if param_value.shape == current_weights[param_name].shape:
                                    compatible_weights[param_name] = param_value
                                else:
                                    print(f"[Client {self.client_id}] Skipping {param_name}: shape mismatch {param_value.shape} vs {current_weights[param_name].shape}")
                            else:
                                compatible_weights[param_name] = param_value
                        else:
                            print(f"[Client {self.client_id}] Skipping {param_name}: not found in current model")
                    
                    if len(compatible_weights) < len(new_sd):
                        print(f"[Client {self.client_id}] Applied {len(compatible_weights)}/{len(new_sd)} compatible parameters")
                    
                    self.trainer.set_weights(compatible_weights)
                
            updated_dict = self.trainer.get_weights()
            updated_params = [updated_dict[k] for k in self.param_keys]
            
            # Get actual training count for initialization phase too
            actual_training_cases = self._get_actual_training_count()
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "initialization_complete": True,
                "actual_training_cases": actual_training_cases
            }
            return updated_params, actual_training_cases, metrics

        # Regular training rounds (federated_round >= 0)
        print(f"[Client {self.client_id}] Training round {federated_round}")
        
        if self.param_keys is None:
            local_sd = self.trainer.get_weights()
            self.param_keys = list(local_sd.keys())

        # Convert list->dict and apply received parameters
        if parameters:
            new_sd = {}
            for k, arr in zip(self.param_keys, parameters):
                new_sd[k] = arr
            
            # Apply FedMA-aware parameter loading
            fedma_enabled = config.get("fedma_enabled", True)
            
            if fedma_enabled:
                # First try to load parameters directly to see if they're compatible
                compatible_direct = self._check_direct_compatibility(new_sd)
                
                if compatible_direct:
                    print(f"[Client {self.client_id}] Parameters are directly compatible, skipping FedMA adaptation")
                    self.trainer.set_weights(new_sd)
                else:
                    print(f"[Client {self.client_id}] Incompatible parameters detected")
                    # Check if we should attempt FedMA adaptation or use safer fallback
                    if self._should_attempt_fedma_adaptation(new_sd):
                        print(f"[Client {self.client_id}] Attempting FedMA adaptation")
                        adapted_weights = self._adapt_fedma_parameters(new_sd, config)
                        self.trainer.set_weights(adapted_weights)
                    else:
                        print(f"[Client {self.client_id}] Using safe parameter filtering instead of FedMA")
                        self._apply_safe_parameter_loading(new_sd)
            else:
                # Traditional compatible parameter loading
                current_weights = self.trainer.get_weights()
                compatible_weights = {}
                
                for param_name, param_value in new_sd.items():
                    if param_name in current_weights:
                        # Check if shapes are compatible
                        if hasattr(param_value, 'shape') and hasattr(current_weights[param_name], 'shape'):
                            if param_value.shape == current_weights[param_name].shape:
                                compatible_weights[param_name] = param_value
                            else:
                                print(f"[Client {self.client_id}] Skipping {param_name}: shape mismatch {param_value.shape} vs {current_weights[param_name].shape}")
                        else:
                            compatible_weights[param_name] = param_value
                    else:
                        print(f"[Client {self.client_id}] Skipping {param_name}: not found in current model")
                
                if len(compatible_weights) < len(new_sd):
                    print(f"[Client {self.client_id}] Applied {len(compatible_weights)}/{len(new_sd)} compatible parameters")
                
                self.trainer.set_weights(compatible_weights)

        # Local training - use minimal epochs for testing
        local_epochs = config.get("local_epochs", 1)  # Reduced from self.local_epochs_per_round for faster testing
        self.trainer.run_training_round(local_epochs)

        updated_dict = self.trainer.get_weights()
        updated_params = [updated_dict[k] for k in self.param_keys]

        # Get actual number of training cases from the trainer's split
        actual_training_cases = self._get_actual_training_count()
        print(f"[Client {self.client_id}] Using {actual_training_cases} training cases for FedAvg aggregation")

        # Training metrics
        final_loss = (
            self.trainer.all_train_losses[-1] if self.trainer.all_train_losses else 0.0
        )

        metrics = {
            "client_id": self.client_id,
            "loss": final_loss,
            "federated_round": federated_round,
            "local_epochs_completed": local_epochs,
            "actual_training_cases": actual_training_cases
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
