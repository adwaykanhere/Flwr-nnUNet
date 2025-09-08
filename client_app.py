# client_app.py

import os
import json
import numpy as np
from typing import Dict
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays

from task import FedNnUNetTrainer
from wandb_integration import get_wandb_logger
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
        
        # Extract dataset name from dataset_json or fallback to folder name
        dataset_name = dataset_dict.get("name", os.path.basename(os.path.dirname(dataset_json)))
        self.dataset_name = dataset_name
        
        # Try to extract modality from dataset metadata
        modality = None
        if "modality" in dataset_dict:
            modality = dataset_dict["modality"]
        elif "channel_names" in dataset_dict and dataset_dict["channel_names"]:
            # Infer modality from channel names if available
            channel_names = dataset_dict["channel_names"]
            if isinstance(channel_names, dict) and "0" in channel_names:
                modality = channel_names["0"]
            elif isinstance(channel_names, list) and len(channel_names) > 0:
                modality = channel_names[0]
        self.modality = modality
        
        self.trainer = FedNnUNetTrainer(
            plans=plans_dict,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_dict,
            device=torch.device("cuda:0"),
        )
        
        # Setup wandb logging with federated parameters
        self.trainer.setup_wandb_logging(
            client_id=client_id,
            dataset_name=dataset_name,
            modality=modality
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
        
        # Initialize wandb logger for client-level logging
        self.wandb_logger = get_wandb_logger(
            run_type="client",
            client_id=client_id,
            dataset_name=dataset_name,
            modality=modality,
            project_suffix="client"
        )
        
        # Initialize warmup and training state
        self.is_warmed_up = False
        self.warmup_epochs = int(os.environ.get('WARMUP_EPOCHS', 2))
        
        # Log initial client configuration
        if self.wandb_logger.enabled:
            client_config = {
                "client/id": client_id,
                "client/dataset_name": dataset_name,
                "client/modality": modality,
                "client/configuration": configuration,
                "client/fold": fold,
                "client/max_epochs": max_total_epochs,
                "client/local_epochs_per_round": local_epochs_per_round,
                "client/num_training_cases": self.num_training_cases
            }
            self.wandb_logger.log_metrics(client_config)

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

    def _get_architecture_signature(self) -> str:
        """
        Generate a unique signature for this client's architecture.
        Used for compatibility checking during aggregation.
        """
        arch_info = self._get_architecture_info()
        input_channels = arch_info.get('input_channels', 1)
        num_classes = arch_info.get('num_classes', 2)
        patch_size = arch_info.get('patch_size', [160, 160, 160])
        
        # Create a deterministic signature
        signature_parts = [
            f"in{input_channels}",
            f"out{num_classes}", 
            f"patch{patch_size[0]}x{patch_size[1]}x{patch_size[2]}" if len(patch_size) >= 3 else f"patch{patch_size}"
        ]
        
        return "_".join(signature_parts)

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
                max_label = 0
                for label_name, label_value in labels.items():
                    if isinstance(label_value, (int, float)):
                        max_label = max(max_label, int(label_value))
                    elif isinstance(label_value, list) and len(label_value) > 0:
                        for x in label_value:
                            if isinstance(x, (int, float)):
                                max_label = max(max_label, int(x))
                
                arch_info["num_classes"] = max_label + 1 if max_label > 0 else 2
            else:
                arch_info["num_classes"] = 2  # Default binary segmentation
            
            # Get patch size from trainer if available
            if hasattr(self.trainer, 'was_initialized') and self.trainer.was_initialized:
                if hasattr(self.trainer, 'configuration_manager'):
                    patch_size = getattr(self.trainer.configuration_manager, 'patch_size', None)
                    if patch_size:
                        arch_info["patch_size"] = str(list(patch_size))
            
            print(f"[Client {self.client_id}] Architecture info: {arch_info}")
            
        except Exception as exc:
            print(f"[Client {self.client_id}] Error extracting architecture info: {exc}")
            arch_info = {
                "input_channels": 1,
                "num_classes": 2,
                "patch_size": "[20, 160, 160]"  # Default 3D patch size
            }
        
        return arch_info

    def _filter_common_backbone_parameters(self, weights_dict: dict) -> dict:
        """
        Filter parameters to only include common backbone layers.
        Uses simple exclusion rules following FednnUNet reference implementation.
        Returns: {param_name: parameter} - using original parameter names
        """
        backbone_dict = {}
        excluded_layers = []
        
        for param_name, param_tensor in weights_dict.items():
            # Check if this is a common backbone layer using simple rules
            if self._is_common_backbone_layer(param_name):
                backbone_dict[param_name] = param_tensor
            else:
                excluded_layers.append((param_name, "ARCHITECTURE_SPECIFIC", param_tensor.shape if hasattr(param_tensor, 'shape') else 'unknown'))
        
        print(f"[Client {self.client_id}] Common backbone filtering: {len(backbone_dict)}/{len(weights_dict)} parameters")
        print(f"[Client {self.client_id}] Excluded {len(excluded_layers)} architecture-specific parameters")
        
        # COMPREHENSIVE LOGGING for debugging parameter filtering
        print(f"\n[Client {self.client_id}] ===== PARAMETER FILTERING DETAILS =====")
        
        # Log all excluded parameters with detailed reasons
        if excluded_layers:
            print(f"[Client {self.client_id}] EXCLUDED parameters ({len(excluded_layers)}):")
            for param_name, reason, shape in excluded_layers:
                is_input = self._is_input_layer(param_name)
                is_output = self._is_output_layer(param_name)
                exclusion_detail = []
                if is_input: exclusion_detail.append("INPUT_LAYER")
                if is_output: exclusion_detail.append("OUTPUT_LAYER")
                if not is_input and not is_output: exclusion_detail.append("NOT_BACKBONE")
                print(f"[Client {self.client_id}]   ❌ {param_name} -> {shape} ({'/'.join(exclusion_detail)})")
        
        # Log all included parameters with their categorization
        if backbone_dict:
            print(f"[Client {self.client_id}] INCLUDED parameters ({len(backbone_dict)}):")
            for param_name, param_tensor in backbone_dict.items():
                shape = param_tensor.shape if hasattr(param_tensor, 'shape') else 'unknown'
                category = "UNKNOWN"
                if "stages.1" in param_name or "stages.2" in param_name: category = "MIDDLE_ENCODER"
                elif "stages.3" in param_name or "stages.4" in param_name: category = "DEEP_ENCODER"
                elif "decoder.stages" in param_name: category = "DECODER_BACKBONE"
                elif "transpconv" in param_name: category = "TRANSPOSE_CONV"
                elif self._is_batch_norm_layer(param_name): category = "BATCH_NORM"
                print(f"[Client {self.client_id}]   ✅ {param_name} -> {shape} ({category})")
        
        print(f"[Client {self.client_id}] =======================================\n")
        
        # Final validation: ensure we have at least some parameters
        if len(backbone_dict) == 0:
            print(f"[Client {self.client_id}] CRITICAL ERROR: No backbone parameters identified!")
            print(f"[Client {self.client_id}] This will cause ClientAppOutputs error.")
        
        return backbone_dict

    def _filter_backbone_parameters(self, weights_dict: dict) -> dict:
        """
        Legacy method for backward compatibility - now delegates to simplified implementation.
        """
        return self._filter_common_backbone_parameters(weights_dict)
    
    def _is_input_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to an input layer that's sensitive to input channels."""
        # nnUNet input layer patterns - comprehensive list including various wrappers
        input_patterns = [
            "_orig_mod.encoder.stages.0.0.convs.0.conv.weight",  # Main nnUNet input layer (torch.compile)
            "_orig_mod.encoder.stages.0.0.convs.0.conv.bias",    # Input layer bias
            "encoder.stages.0.0.convs.0.conv.weight",  # Without torch.compile wrapper
            "encoder.stages.0.0.convs.0.conv.bias",    # Input layer bias (no wrapper)
            "network.encoder.stages.0.0.convs.0.conv.weight",  # Alternative wrapper
            "network.encoder.stages.0.0.convs.0.conv.bias",    # Alternative wrapper bias
            "model.encoder.stages.0.0.convs.0.conv.weight",    # Model wrapper
            "model.encoder.stages.0.0.convs.0.conv.bias",      # Model wrapper bias
            # Additional first stage patterns
            "_orig_mod.encoder.stages.0.1.convs.0.conv.weight",  # First stage second block
            "_orig_mod.encoder.stages.0.1.convs.0.conv.bias",
            "encoder.stages.0.1.convs.0.conv.weight",
            "encoder.stages.0.1.convs.0.conv.bias",
        ]
        
        # Also check for first encoder stage patterns (broader match for safety)
        first_stage_patterns = [
            "encoder.stages.0.0",  # First convolution block
            "_orig_mod.encoder.stages.0.0",  # With wrapper
            "network.encoder.stages.0.0",   # Alternative wrapper
            "model.encoder.stages.0.0",     # Model wrapper
        ]
        
        # Check exact matches first
        if param_name in input_patterns:
            return True
            
        # Check for first stage patterns with conv layers
        for pattern in first_stage_patterns:
            if pattern in param_name and "convs.0.conv" in param_name and (".weight" in param_name or ".bias" in param_name):
                return True
                
        return False
    
    def _is_output_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to an output layer that's sensitive to number of classes."""
        # nnUNet output layer patterns - comprehensive list including various wrappers
        output_patterns = [
            "_orig_mod.decoder.seg_layers",  # Main segmentation layers (torch.compile)
            "decoder.seg_layers",  # Without torch.compile wrapper
            "_orig_mod.seg_layers",  # Alternative structure
            "seg_layers",  # Direct segmentation layers
            "network.decoder.seg_layers",  # Network wrapper
            "model.decoder.seg_layers",    # Model wrapper
            "network.seg_layers",          # Network wrapper alternative
            "model.seg_layers",            # Model wrapper alternative
            # Additional output patterns
            "output_layer", "final_layer", "classifier", "head"
        ]
        
        # Also check for final output layer patterns (typically the last segmentation layer)
        final_output_patterns = [
            "seg_layers.4",  # Common final segmentation layer index
            "seg_layers.3",  # Alternative final layer
            "seg_layers.2",  # Alternative final layer
            "final_layer",   # Explicit final layer naming
            "output_layer",  # Explicit output layer naming
        ]
        
        # Check if parameter name contains any output pattern
        for pattern in output_patterns:
            if pattern in param_name and (".weight" in param_name or ".bias" in param_name):
                return True
        
        # Check for final output patterns
        for pattern in final_output_patterns:
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

    def _has_architecture_dependent_shape(self, param_name: str, param_tensor) -> bool:
        """Check if parameter has architecture-dependent shape based on known patterns."""
        if not hasattr(param_tensor, 'shape'):
            return False
            
        shape = param_tensor.shape
        
        # Input layer heuristics: typically has shape related to input channels
        # Common nnUNet input layer shapes: (out_channels, in_channels, kernel...)
        if len(shape) >= 2:
            # Check if this looks like an input convolution (small input channel dimension)
            in_channels = shape[1] if len(shape) > 1 else shape[0]
            if in_channels <= 5:  # Typical medical imaging: 1-4 input channels
                # Additional check: parameter name suggests input layer
                input_indicators = ["stages.0.0", "convs.0.conv", "first", "input"]
                if any(indicator in param_name.lower() for indicator in input_indicators):
                    return True
        
        # Output layer heuristics: typically has shape related to number of classes
        # Common nnUNet output layer shapes: (num_classes, features, kernel...)
        if len(shape) >= 1:
            # Check if this looks like an output layer (small output channel dimension)
            out_channels = shape[0]
            if out_channels <= 20:  # Typical segmentation: 2-10 classes
                # Additional check: parameter name suggests output layer
                output_indicators = ["seg_layers", "final", "output", "classifier"]
                if any(indicator in param_name.lower() for indicator in output_indicators):
                    return True
        
        return False

    def _convert_to_semantic_name(self, param_name: str) -> str:
        """
        Convert PyTorch parameter names to semantic names that can be matched across architectures.
        This enables common layer detection across heterogeneous client architectures.
        """
        # Remove common wrappers first
        semantic_name = param_name
        wrappers = ["_orig_mod.", "network.", "model."]
        for wrapper in wrappers:
            if semantic_name.startswith(wrapper):
                semantic_name = semantic_name[len(wrapper):]
                break
        
        # Handle encoder layers
        if "encoder.stages." in semantic_name:
            # Pattern: encoder.stages.0.0.convs.0.conv.weight -> enc_s0_b0_conv0_w
            parts = semantic_name.split(".")
            try:
                stage_idx = None
                block_idx = None
                conv_idx = None
                param_type = None
                
                for i, part in enumerate(parts):
                    if part == "encoder" and i + 2 < len(parts) and parts[i + 1] == "stages":
                        stage_idx = parts[i + 2]
                        if i + 3 < len(parts):
                            block_idx = parts[i + 3]
                    elif part == "convs" and i + 1 < len(parts):
                        conv_idx = parts[i + 1]
                    elif part in ["weight", "bias"]:
                        param_type = "w" if part == "weight" else "b"
                    elif part == "conv":
                        continue  # Skip conv identifier
                    elif part in ["norm", "bn", "batch_norm"]:
                        param_type = "norm_" + (param_type or "w")
                
                if stage_idx is not None and block_idx is not None:
                    semantic_parts = [f"enc_s{stage_idx}_b{block_idx}"]
                    if conv_idx is not None:
                        semantic_parts.append(f"conv{conv_idx}")
                    if param_type:
                        semantic_parts.append(param_type)
                    return "_".join(semantic_parts)
            except (IndexError, ValueError):
                pass
        
        # Handle decoder layers (excluding seg_layers which are architecture-specific)
        if "decoder." in semantic_name and "seg_layers" not in semantic_name:
            parts = semantic_name.split(".")
            try:
                if "transpconvs" in semantic_name:
                    # Pattern: decoder.transpconvs.0.weight -> dec_transpconv0_w
                    for i, part in enumerate(parts):
                        if part == "transpconvs" and i + 1 < len(parts):
                            layer_idx = parts[i + 1]
                            param_type = "w" if semantic_name.endswith("weight") else "b"
                            return f"dec_transpconv{layer_idx}_{param_type}"
                else:
                    # Regular decoder layers
                    stage_idx = None
                    block_idx = None
                    conv_idx = None
                    param_type = None
                    
                    for i, part in enumerate(parts):
                        if part == "stages" and i + 1 < len(parts):
                            stage_idx = parts[i + 1]
                            if i + 2 < len(parts):
                                block_idx = parts[i + 2]
                        elif part == "convs" and i + 1 < len(parts):
                            conv_idx = parts[i + 1]
                        elif part in ["weight", "bias"]:
                            param_type = "w" if part == "weight" else "b"
                        elif part in ["norm", "bn"]:
                            param_type = "norm_" + (param_type or "w")
                    
                    if stage_idx is not None:
                        semantic_parts = [f"dec_s{stage_idx}"]
                        if block_idx is not None:
                            semantic_parts.append(f"b{block_idx}")
                        if conv_idx is not None:
                            semantic_parts.append(f"conv{conv_idx}")
                        if param_type:
                            semantic_parts.append(param_type)
                        return "_".join(semantic_parts)
            except (IndexError, ValueError):
                pass
        
        # Skip architecture-dependent layers entirely
        if self._is_input_layer(param_name) or self._is_output_layer(param_name):
            return None  # Don't include in common parameters
        
        # Fallback: create a simplified semantic name for unmatched patterns
        # Remove common suffixes and normalize
        simplified = semantic_name.replace(".", "_")
        simplified = simplified.replace("weight", "w").replace("bias", "b")
        
        # If it's clearly a backbone layer, return simplified name
        if any(indicator in simplified.lower() for indicator in ["encoder", "decoder", "conv", "norm", "bn"]):
            return f"backbone_{simplified}"
        
        # If we can't classify it, return None to exclude it
        return None

    def _is_common_backbone_layer(self, param_name: str) -> bool:
        """
        Check if a parameter belongs to the common backbone (shared across architectures).
        Uses a whitelist approach to identify definitely shareable layers.
        """
        # WHITELIST APPROACH: Only include layers we're confident are shareable
        
        # 1. EXCLUDE input layers (architecture-specific)
        if self._is_input_layer(param_name):
            return False
            
        # 2. EXCLUDE output layers (architecture-specific)  
        if self._is_output_layer(param_name):
            return False
        
        # 3. INCLUDE ONLY middle encoder layers (stages 1-3, EXCLUDE stage 0 and final stages)
        middle_encoder_patterns = [
            "encoder.stages.1", "encoder.stages.2", "encoder.stages.3",
            "_orig_mod.encoder.stages.1", "_orig_mod.encoder.stages.2", 
            "_orig_mod.encoder.stages.3",
            "network.encoder.stages.1", "network.encoder.stages.2",
            "network.encoder.stages.3",
            "model.encoder.stages.1", "model.encoder.stages.2",
            "model.encoder.stages.3"
        ]
        
        for pattern in middle_encoder_patterns:
            if pattern in param_name:
                return True
        
        # 4. INCLUDE ONLY middle decoder layers (excluding first decoder stage and seg_layers)
        decoder_backbone_patterns = [
            "decoder.stages.1", "decoder.stages.2", "decoder.stages.3",
            "_orig_mod.decoder.stages.1", "_orig_mod.decoder.stages.2", 
            "_orig_mod.decoder.stages.3",
            "network.decoder.stages.1", "network.decoder.stages.2",
            "network.decoder.stages.3",
            "model.decoder.stages.1", "model.decoder.stages.2",
            "model.decoder.stages.3",
            # Transpose convolutions (middle layers only)
            "decoder.transpconvs.1", "decoder.transpconvs.2", "decoder.transpconvs.3",
            "_orig_mod.decoder.transpconvs.1", "_orig_mod.decoder.transpconvs.2",
            "_orig_mod.decoder.transpconvs.3"
        ]
        
        for pattern in decoder_backbone_patterns:
            if pattern in param_name and "seg_layers" not in param_name:
                return True
        
        # 5. INCLUDE batch normalization layers (always shareable)
        if self._is_batch_norm_layer(param_name):
            return True
        
        # 6. INCLUDE ONLY middle convolution layers (stages 1-3, NOT first or last)
        if ("conv" in param_name.lower() and 
            not self._is_input_layer(param_name) and 
            not self._is_output_layer(param_name) and
            any(stage in param_name for stage in ["stages.1", "stages.2", "stages.3"]) and
            "stages.0" not in param_name):  # Explicitly exclude stage 0 (first layer)
            return True
        
        # 7. DEFAULT: Exclude everything else to be safe
        return False

    def _log_parameter_structure(self, weights_dict: dict, context: str):
        """Log detailed parameter structure for debugging."""
        print(f"\n[Client {self.client_id}] ===== 3D MODEL ARCHITECTURE ({context}) =====")
        print(f"[Client {self.client_id}] Total parameters: {len(weights_dict)}")
        print(f"[Client {self.client_id}] Architecture: 3D nnUNet")
        
        # Extract architecture info
        arch_info = self._get_architecture_info()
        print(f"[Client {self.client_id}] Input channels: {arch_info.get('input_channels', 'unknown')}")
        print(f"[Client {self.client_id}] Output classes: {arch_info.get('num_classes', 'unknown')}")
        print(f"[Client {self.client_id}] Patch size: {arch_info.get('patch_size', 'unknown')}")
        
        # Categorize parameters by type and mark exclusion status
        encoder_params = []
        decoder_backbone_params = []
        seg_layer_params = []
        transpconv_params = []
        other_params = []
        
        for param_name, param_tensor in weights_dict.items():
            shape_str = f"{param_tensor.shape}" if hasattr(param_tensor, 'shape') else "unknown"
            
            # Determine exclusion status for this parameter
            exclusion_type = ""
            if self._is_input_layer(param_name):
                exclusion_type = "INPUT_LAYER"
            elif self._is_output_layer(param_name):
                exclusion_type = "OUTPUT_LAYER"
            elif self._has_architecture_dependent_shape(param_name, param_tensor):
                exclusion_type = "ARCHITECTURE_DEPENDENT"
            else:
                exclusion_type = "BACKBONE" if "_orig_mod.decoder" in param_name or "decoder" in param_name else "ENCODER"
                if "_orig_mod.decoder.transpconvs" in param_name or "transpconvs" in param_name:
                    exclusion_type = "TRANSPCONV"
            
            param_info = (param_name, shape_str, exclusion_type)
            
            if "_orig_mod.encoder" in param_name or ("encoder" in param_name and "decoder.encoder" not in param_name):
                encoder_params.append(param_info)
            elif "_orig_mod.decoder.seg_layers" in param_name or "seg_layers" in param_name:
                seg_layer_params.append(param_info)
            elif "_orig_mod.decoder.transpconvs" in param_name or "transpconvs" in param_name:
                transpconv_params.append(param_info)
            elif "_orig_mod.decoder" in param_name or "decoder" in param_name:
                decoder_backbone_params.append(param_info)
            else:
                other_params.append(param_info)
        
        print(f"[Client {self.client_id}] Encoder parameters: {len(encoder_params)}")
        if encoder_params:
            first = encoder_params[0]
            last = encoder_params[-1]
            print(f"[Client {self.client_id}]   First: {first[0]} -> {first[1]} ({first[2]})")
            print(f"[Client {self.client_id}]   Last:  {last[0]} -> {last[1]} ({last[2]})")
        
        print(f"[Client {self.client_id}] Decoder backbone parameters: {len(decoder_backbone_params)}")
        if decoder_backbone_params:
            first = decoder_backbone_params[0]
            last = decoder_backbone_params[-1]
            print(f"[Client {self.client_id}]   First: {first[0]} -> {first[1]} ({first[2]})")
            print(f"[Client {self.client_id}]   Last:  {last[0]} -> {last[1]} ({last[2]})")
        
        print(f"[Client {self.client_id}] Transpose convolution parameters: {len(transpconv_params)}")
        for param_name, shape_str, exc_type in transpconv_params:
            print(f"[Client {self.client_id}]   {param_name} -> {shape_str} ({exc_type})")
        
        print(f"[Client {self.client_id}] Segmentation layer parameters: {len(seg_layer_params)}")
        for param_name, shape_str, exc_type in seg_layer_params:
            print(f"[Client {self.client_id}]   {param_name} -> {shape_str} ({exc_type})")
        
        if other_params:
            print(f"[Client {self.client_id}] Other parameters: {len(other_params)}")
            for param_name, shape_str, exc_type in other_params[:3]:  # Show first 3
                print(f"[Client {self.client_id}]   {param_name} -> {shape_str} ({exc_type})")
        
        print(f"[Client {self.client_id}] ============================================\n")

    def _load_backbone_parameters(self, backbone_parameters: list):
        """Load backbone parameters from the server, updating only matching pairs.

        Any local parameters without a corresponding name from the server remain
        unchanged.
        """
        try:
            current_weights = self.trainer.get_weights()
            updated_weights = current_weights.copy()

            param_names = getattr(self, "param_names", [])

            if len(backbone_parameters) != len(param_names):
                print(
                    f"[Client {self.client_id}] Warning: received {len(backbone_parameters)} parameters for {len(param_names)} names"
                )

            attempted = min(len(backbone_parameters), len(param_names))
            print(f"[Client {self.client_id}] Loading {attempted} backbone parameters from server")

            loaded_count = 0
            skipped_count = 0
            shape_mismatch_count = 0

            for param_name, param_value in zip(param_names, backbone_parameters):
                if param_name in current_weights:
                    if hasattr(param_value, "shape") and hasattr(current_weights[param_name], "shape"):
                        if param_value.shape == current_weights[param_name].shape:
                            updated_weights[param_name] = param_value
                            loaded_count += 1
                        else:
                            print(
                                f"[Client {self.client_id}] Shape mismatch for {param_name}: server {param_value.shape} vs local {current_weights[param_name].shape}"
                            )
                            shape_mismatch_count += 1
                    else:
                        updated_weights[param_name] = param_value
                        loaded_count += 1
                else:
                    print(
                        f"[Client {self.client_id}] Parameter {param_name} not found in local model, skipping"
                    )
                    skipped_count += 1

            try:
                self.trainer.set_weights(updated_weights)
                print(
                    f"[Client {self.client_id}] Successfully loaded parameters into model with strict=False"
                )
            except Exception as set_weights_error:
                print(
                    f"[Client {self.client_id}] Warning: Error in set_weights: {set_weights_error}"
                )
                print(
                    f"[Client {self.client_id}] This may be due to remaining architecture mismatches"
                )
                print(
                    f"[Client {self.client_id}] Continuing with current local parameters"
                )
                return

            print(f"[Client {self.client_id}] ===== PARAMETER LOADING SUMMARY =====")
            print(
                f"[Client {self.client_id}]   Successfully loaded: {loaded_count} parameters"
            )
            print(
                f"[Client {self.client_id}]   Skipped (not found locally): {skipped_count} parameters"
            )
            print(
                f"[Client {self.client_id}]   Skipped (shape mismatch): {shape_mismatch_count} parameters"
            )
            print(
                f"[Client {self.client_id}]   Total parameters attempted: {attempted}"
            )

            if loaded_count == 0:
                print(
                    f"[Client {self.client_id}] WARNING: No parameters were loaded! Check parameter compatibility."
                )
            elif shape_mismatch_count > 0 or skipped_count > 0:
                print(
                    f"[Client {self.client_id}] INFO: Some parameters skipped - this is normal for heterogeneous architectures"
                )

        except Exception as e:
            print(
                f"[Client {self.client_id}] ERROR: Exception during backbone parameter loading: {e}"
            )
            print(
                f"[Client {self.client_id}] Keeping current local parameters as fallback"
            )
            import traceback
            print(
                f"[Client {self.client_id}] Traceback: {traceback.format_exc()}"
            )

    
    
    
    
    
    

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
        Return current model parameters as a simple list of numpy arrays (Flower-compatible).
        For federated nnUNet: only return common backbone layers.
        Raw parameter names and metadata will be passed through fit() metrics.
        """
        if not self.trainer.was_initialized:
            self.trainer.initialize()

        weights_dict = self.trainer.get_weights()
        
        # Log detailed parameter structure for debugging
        self._log_parameter_structure(weights_dict, "filter_common_backbone_parameters")
        
        # Filter to common backbone parameters (using raw parameter names)
        backbone_dict = self._filter_common_backbone_parameters(weights_dict)
        
        if not backbone_dict:
            print(f"[Client {self.client_id}] WARNING: No common backbone parameters found after strict filtering!")
            print(f"[Client {self.client_id}] Applying minimum parameter guarantee...")
            
            # MINIMUM PARAMETER GUARANTEE: Include at least some middle layers
            fallback_backbone_dict = {}
            
            # Try to include ONLY middle encoder layers as fallback (NO first/last layers)
            for param_name, param_tensor in weights_dict.items():
                if (any(stage in param_name for stage in ["stages.1", "stages.2", "stages.3"]) and
                    "conv" in param_name.lower() and
                    "stages.0" not in param_name and  # Explicitly exclude first stage
                    "stages.4" not in param_name and  # Explicitly exclude potential last stage
                    not self._is_input_layer(param_name) and
                    not self._is_output_layer(param_name)):
                    fallback_backbone_dict[param_name] = param_tensor
            
            if fallback_backbone_dict:
                print(f"[Client {self.client_id}] Found {len(fallback_backbone_dict)} fallback backbone parameters")
                backbone_dict = fallback_backbone_dict
            else:
                print(f"[Client {self.client_id}] CRITICAL: No parameters can be safely shared! Using empty set.")
                self.param_names = []
                return []
        
        # Store parameter names in sorted order for consistency across clients
        self.param_names = sorted(backbone_dict.keys())
        
        # Return parameters in consistent sorted order (Flower-compatible NDArrays)
        param_values = [backbone_dict[name] for name in self.param_names]
        
        print(f"[Client {self.client_id}] Sending {len(self.param_names)} common backbone parameters")
        print(f"[Client {self.client_id}] Parameter names preview: {self.param_names[:3]}..." if len(self.param_names) > 3 else f"[Client {self.client_id}] Parameter names: {self.param_names}")
        
        return param_values

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
        # Update parameter name list from server configuration if provided
        param_names_str = config.get("param_names_str")
        if param_names_str:
            try:
                self.param_names = json.loads(param_names_str)
            except json.JSONDecodeError as e:
                print(
                    f"[Client {self.client_id}] Warning: Could not decode param_names_str: {e}"
                )

        federated_round = config.get("server_round", 1)
        
        # Handle preprocessing round (federated_round = -2) - share fingerprint only
        if federated_round == -2:
            print(f"[Client {self.client_id}] Preprocessing round - sharing fingerprint")
            if not self.trainer.was_initialized:
                self.trainer.initialize()
            
            # Get initial common backbone parameters (simple format)
            initial_weights = self.trainer.get_weights()
            backbone_dict = self._filter_common_backbone_parameters(initial_weights)
            self.param_names = sorted(backbone_dict.keys())  # Store sorted parameter names
            
            # Return parameters in simple list format
            initial_params = [backbone_dict[name] for name in self.param_names]
            
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
                "is_warmup": False,
                # Parameter metadata for common layer aggregation
                "param_names_str": json.dumps(self.param_names),
                "param_shapes_str": json.dumps([list(initial_params[i].shape) for i in range(len(initial_params))]) if initial_params else "[]",
                "num_params": len(initial_params),
                # Architecture compatibility signature
                "architecture_signature": self._get_architecture_signature()
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
                
            # Initialize param_names if needed
            if not hasattr(self, 'param_names') or not self.param_names:
                local_sd = self.trainer.get_weights()
                backbone_dict = self._filter_common_backbone_parameters(local_sd)
                self.param_names = sorted(backbone_dict.keys())
                
            # Apply received global backbone parameters (simple list format)
            if parameters:
                # Load backbone parameters while preserving architecture-specific layers
                self._load_backbone_parameters(parameters)
                
            # Return updated parameters in simple format
            updated_dict = self.trainer.get_weights()
            backbone_dict = self._filter_common_backbone_parameters(updated_dict)
            
            # Return parameters in consistent sorted order
            updated_params = [backbone_dict[name] for name in self.param_names if name in backbone_dict]
            
            # SAFETY CHECK: Ensure non-empty parameter list
            if not updated_params:
                print(f"[Client {self.client_id}] ERROR: Initialization resulted in empty parameter list!")
                print(f"[Client {self.client_id}] param_names: {self.param_names}")
                print(f"[Client {self.client_id}] backbone_dict keys: {list(backbone_dict.keys())}")
                raise RuntimeError("Empty parameter list would cause ClientAppOutputs error")
            
            # Get actual training count for initialization phase too
            actual_training_cases = self._get_actual_training_count()
            
            metrics = {
                "client_id": self.client_id,
                "loss": 0.0,
                "initialization_complete": True,
                "actual_training_cases": actual_training_cases,
                "is_warmup": False,
                # Parameter metadata for common layer aggregation
                "param_names_str": json.dumps(self.param_names),
                "param_shapes_str": json.dumps([list(updated_params[i].shape) for i in range(len(updated_params))]) if updated_params else "[]",
                "num_params": len(updated_params),
                # Architecture compatibility signature
                "architecture_signature": self._get_architecture_signature()
            }
            return updated_params, actual_training_cases, metrics

        # Regular training rounds (federated_round >= 0)
        print(f"[Client {self.client_id}] Training round {federated_round}")
        
        # Log round start to wandb
        if self.wandb_logger.enabled:
            self.wandb_logger.log_metrics({
                "federated/round": federated_round,
                "federated/round_start": True
            }, step=federated_round)
        
        # Check if this is round 0 (warmup round)
        is_warmup_round = (federated_round == 0)
        
        if not hasattr(self, 'param_names') or not self.param_names:
            local_sd = self.trainer.get_weights()
            backbone_dict = self._filter_common_backbone_parameters(local_sd)
            self.param_names = sorted(backbone_dict.keys())

        # Handle warmup logic for round 0
        if is_warmup_round and not self.is_warmed_up:
            print(f"[Client {self.client_id}] Round 0: Starting warmup phase")
            # Warm up first and last layers locally
            self._warmup_first_last_layers(self.warmup_epochs)
            
            # No parameter loading from server in warmup round
            # Just return current backbone parameters after warmup (simple format)
            updated_dict = self.trainer.get_weights()
            backbone_dict = self._filter_common_backbone_parameters(updated_dict)
            
            # Return parameters in consistent sorted order
            updated_params = [backbone_dict[name] for name in self.param_names if name in backbone_dict]
            
            # SAFETY CHECK: Ensure non-empty parameter list
            if not updated_params:
                print(f"[Client {self.client_id}] ERROR: Warmup resulted in empty parameter list!")
                print(f"[Client {self.client_id}] param_names: {self.param_names}")
                print(f"[Client {self.client_id}] backbone_dict keys: {list(backbone_dict.keys())}")
                raise RuntimeError("Empty parameter list would cause ClientAppOutputs error")
            
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
                "warmup_complete": True,
                # Parameter metadata for common layer aggregation
                "param_names_str": json.dumps(self.param_names),
                "param_shapes_str": json.dumps([list(updated_params[i].shape) for i in range(len(updated_params))]) if updated_params else "[]",
                "num_params": len(updated_params),
                # Architecture compatibility signature
                "architecture_signature": self._get_architecture_signature()
            }
            
            # Log warmup completion to wandb
            if self.wandb_logger.enabled:
                warmup_metrics = {
                    "federated/warmup_loss": final_loss,
                    "federated/warmup_epochs": self.warmup_epochs,
                    "federated/warmup_complete": True,
                    "federated/training_cases": actual_training_cases
                }
                self.wandb_logger.log_metrics(warmup_metrics, step=federated_round)
            
            return updated_params, actual_training_cases, metrics
        
        # Regular training rounds (federated_round > 0)
        # Load backbone parameters from server (simple list format)
        if parameters:
            # Load backbone parameters while preserving architecture-specific layers
            self._load_backbone_parameters(parameters)

        # Local training
        local_epochs = config.get("local_epochs", 1)
        self.trainer.run_training_round(local_epochs)

        # Return updated backbone parameters only (simple format)
        updated_dict = self.trainer.get_weights()
        backbone_dict = self._filter_common_backbone_parameters(updated_dict)
        
        # Return parameters in consistent sorted order
        updated_params = [backbone_dict[name] for name in self.param_names if name in backbone_dict]
        
        # SAFETY CHECK: Ensure non-empty parameter list
        if not updated_params:
            print(f"[Client {self.client_id}] ERROR: Training resulted in empty parameter list!")
            print(f"[Client {self.client_id}] param_names: {self.param_names}")
            print(f"[Client {self.client_id}] backbone_dict keys: {list(backbone_dict.keys())}")
            raise RuntimeError("Empty parameter list would cause ClientAppOutputs error")

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
            "is_warmup": False,
            # Parameter metadata for common layer aggregation
            "param_names_str": json.dumps(self.param_names),
            "param_shapes_str": json.dumps([list(updated_params[i].shape) for i in range(len(updated_params))]) if updated_params else "[]",
            "num_params": len(updated_params),
            # Architecture compatibility signature
            "architecture_signature": self._get_architecture_signature()
        }
        
        # Log training completion to wandb
        if self.wandb_logger.enabled:
            training_metrics = {
                "federated/training_loss": final_loss,
                "federated/local_epochs": local_epochs,
                "federated/training_cases": actual_training_cases,
                "federated/round_complete": True
            }
            self.wandb_logger.log_metrics(training_metrics, step=federated_round)
        
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
                
                # Log validation results to wandb
                if self.wandb_logger.enabled:
                    val_metrics = {
                        "federated/validation_dice_mean": current_dice,
                        "federated/validation_num_batches": validation_results.get('num_batches', 0)
                    }
                    # Log per-class dice scores
                    for label, score in per_label_scores.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            val_metrics[f"federated/validation_dice_class_{label}"] = float(score)
                    
                    self.wandb_logger.log_metrics(val_metrics, step=federated_round)
                
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
                                
                                # Log model artifact to wandb
                                if self.wandb_logger.enabled:
                                    try:
                                        artifact_name = f"client_{self.client_id}_best_model_round_{federated_round}"
                                        artifact_metadata = {
                                            "client_id": self.client_id,
                                            "federated_round": federated_round,
                                            "validation_dice": current_dice,
                                            "is_best": True,
                                            "dataset_name": self.dataset_name,
                                            "modality": self.modality
                                        }
                                        self.wandb_logger.log_model_artifact(
                                            model_path=checkpoint_path,
                                            artifact_name=artifact_name,
                                            artifact_type="federated_model",
                                            metadata=artifact_metadata
                                        )
                                        
                                        # Log checkpoint metrics
                                        checkpoint_metrics = {
                                            "federated/best_model_updated": True,
                                            "federated/best_validation_dice": current_dice,
                                            "federated/checkpoint_round": federated_round
                                        }
                                        self.wandb_logger.log_metrics(checkpoint_metrics, step=federated_round)
                                    except Exception as artifact_error:
                                        print(f"[Client {self.client_id}] Failed to log model artifact to wandb: {artifact_error}")
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
