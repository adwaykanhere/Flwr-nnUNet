# server_app_modality.py

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import flwr as fl
from flwr.common import Context, FitRes, NDArrays, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from task import merge_dataset_fingerprints


class ModalityAwareFederatedStrategy(FedAvg):
    """
    Modality-aware FedAvg strategy that implements multi-phase federated learning for nnUNet:
    - Groups clients by modality (CT, MR, etc.)
    - Performs intra-modality aggregation first
    - Then performs weighted inter-modality aggregation
    - Supports traditional phases: preprocessing, initialization, training
    """

    def __init__(self, 
                 expected_num_clients: int = 2, 
                 enable_modality_aggregation: bool = False,
                 modality_weights: Optional[Dict[str, float]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.expected_num_clients = expected_num_clients
        self.fingerprints_collected: List[dict] = []
        self.global_fingerprint: dict | None = None
        self.current_phase = "preprocessing"  # preprocessing, initialization, training
        
        # Modality-aware aggregation settings
        self.enable_modality_aggregation = enable_modality_aggregation
        self.modality_weights = modality_weights or {}
        self.client_modalities: Dict[str, str] = {}  # client_id -> modality
        self.client_datasets: Dict[str, str] = {}  # client_id -> dataset
        self.client_signatures: Dict[str, str] = {}  # client_id -> dataset_modality signature
        self.modality_groups: Dict[str, List[str]] = {}  # modality -> [client_ids]
        self.dataset_modality_groups: Dict[str, List[str]] = {}  # dataset_modality -> [client_ids]
        
        # Best global model tracking
        self.best_global_validation_dice = 0.0
        self.best_global_round = 0
        self.global_models_dir = "global_models"
        
        # Per-modality model tracking
        self.best_modality_models: Dict[str, Dict] = {}  # modality -> {dice, round, params}
        
        # Warmup tracking for backbone aggregation strategy
        self.client_warmup_status: Dict[str, bool] = {}  # client_id -> is_warmed_up

        self.last_common_param_names: List[str] = []
        
        # Create global models directory
        os.makedirs(self.global_models_dir, exist_ok=True)
        print(f"[Server] Global models will be saved to: {self.global_models_dir}")
        print(f"[Server] Modality-aware aggregation: {'enabled' if self.enable_modality_aggregation else 'disabled'}")
        print(f"[Server] Backbone aggregation strategy: enabled (first/last layers remain local)")
        if self.modality_weights:
            print(f"[Server] Modality weights: {self.modality_weights}")

    def extract_modality_from_metadata(self, metrics: Dict) -> Optional[str]:
        """Extract modality information from client metrics"""
        # Look for modality in various possible keys
        modality_keys = ['modality', 'client_modality', 'dataset_modality']
        
        for key in modality_keys:
            if key in metrics:
                return str(metrics[key]).upper()
        
        # Try to infer from channel names if available
        if 'channel_names' in metrics:
            channel_names = metrics['channel_names']
            
            # Handle both dict and list formats
            if isinstance(channel_names, dict) and len(channel_names) > 0:
                # Get first channel value from dict
                first_channel = str(list(channel_names.values())[0]).lower()
            elif isinstance(channel_names, list) and len(channel_names) > 0:
                # Get first channel from list
                first_channel = str(channel_names[0]).lower()
            else:
                first_channel = ""
                
            if first_channel:
                # Common modality patterns
                if 'ct' in first_channel or 'computed' in first_channel:
                    return 'CT'
                elif 'mr' in first_channel or 'magnetic' in first_channel or 't1' in first_channel or 't2' in first_channel:
                    return 'MR'
                elif 'pet' in first_channel:
                    return 'PET'
                elif 'us' in first_channel or 'ultrasound' in first_channel:
                    return 'US'
        
        # Default fallback
        return 'UNKNOWN'

    def extract_dataset_from_metadata(self, metrics: Dict) -> Optional[str]:
        """Extract dataset information from client metrics"""
        # Look for dataset information
        dataset_keys = ['dataset_id', 'dataset_name']
        
        for key in dataset_keys:
            if key in metrics:
                return str(metrics[key])
        
        # Default fallback
        return 'UNKNOWN'

    def create_client_signature(self, client_id: str, metrics: Dict) -> str:
        """Create a unique signature for client based on dataset and modality"""
        dataset = self.extract_dataset_from_metadata(metrics)
        modality = self.extract_modality_from_metadata(metrics)
        return f"{dataset}_{modality}"

    def update_client_modality_mapping(self, client_id: str, metrics: Dict):
        """Update the mapping of clients to their modalities and datasets"""
        modality = self.extract_modality_from_metadata(metrics)
        dataset = self.extract_dataset_from_metadata(metrics)
        signature = self.create_client_signature(client_id, metrics)
        
        # Update mappings
        if modality != 'UNKNOWN':
            self.client_modalities[client_id] = modality
        if dataset != 'UNKNOWN':
            self.client_datasets[client_id] = dataset
        
        self.client_signatures[client_id] = signature
        
        
        # Update modality groups (traditional)
        if modality not in self.modality_groups:
            self.modality_groups[modality] = []
        if client_id not in self.modality_groups[modality]:
            self.modality_groups[modality].append(client_id)
        
        # Update dataset-modality groups (enhanced for multi-dataset)
        if signature not in self.dataset_modality_groups:
            self.dataset_modality_groups[signature] = []
        if client_id not in self.dataset_modality_groups[signature]:
            self.dataset_modality_groups[signature].append(client_id)
        
        print(f"[Server] Client {client_id} mapping updated:")
        print(f"  Dataset: {dataset}")
        print(f"  Modality: {modality}")
        print(f"  Signature: {signature}")
        

    def aggregate_within_modality(self, 
                                  modality: str, 
                                  client_results: List[Tuple[ClientProxy, FitRes]]) -> Tuple[NDArrays, Dict]:
        """Aggregate parameters within a single modality group"""
        print(f"[Server] Aggregating {len(client_results)} clients within modality: {modality}")
        
        # Extract parameters and weights for this modality
        modality_params_and_weights = []
        total_examples = 0
        modality_metrics = []
        
        for client_proxy, fit_res in client_results:
            client_id = str(fit_res.metrics.get("client_id", "unknown"))
            if self.client_modalities.get(client_id) == modality:
                parameters = parameters_to_ndarrays(fit_res.parameters)
                num_examples = fit_res.num_examples
                modality_params_and_weights.append((parameters, num_examples))
                total_examples += num_examples
                modality_metrics.append(fit_res.metrics)
        
        if not modality_params_and_weights:
            return None, {}
        
        # Use asymmetric aggregation for modality-level aggregation
        if len(modality_params_and_weights) > 1:
            # Extract client results for this modality
            modality_client_results = [(client_proxy, fit_res) for client_proxy, fit_res in client_results 
                                     if self.client_modalities.get(str(fit_res.metrics.get("client_id", "unknown"))) == modality]
            
            if len(modality_client_results) > 1:
                # Use simple common layer aggregation following FednnUNet approach
                client_info = self._extract_client_parameter_info(modality_client_results)
                compatible_params = self._find_common_layers(client_info)
                
                if compatible_params and len(compatible_params) > 0:
                    print(f"[Server] Modality {modality}: aggregating {len(compatible_params)} common parameters")
                    aggregated_param_dict = self._asymmetric_weighted_average(client_info, compatible_params)
                    common_param_names = sorted(compatible_params.keys())
                    aggregated_params = [aggregated_param_dict[name] for name in common_param_names]
                    self.last_common_param_names = common_param_names
                else:
                    print(f"[Server] Modality {modality}: no common parameters found")
                    return None, {}
            else:
                aggregated_params = self._weighted_average(modality_params_and_weights)
        else:
            aggregated_params = self._weighted_average(modality_params_and_weights)
        
        # Calculate modality-specific metrics
        modality_validation_scores = []
        for metrics in modality_metrics:
            validation_dice = metrics.get("validation_dice", {})
            if isinstance(validation_dice, dict) and "mean" in validation_dice:
                modality_validation_scores.append(validation_dice["mean"])
        
        avg_modality_dice = np.mean(modality_validation_scores) if modality_validation_scores else 0.0
        
        modality_summary = {
            'modality': modality,
            'num_clients': len(modality_params_and_weights),
            'total_examples': total_examples,
            'avg_validation_dice': avg_modality_dice,
            'validation_scores': modality_validation_scores
        }
        
        print(f"[Server] Modality {modality}: {len(modality_params_and_weights)} clients, "
              f"{total_examples} examples, avg_dice={avg_modality_dice:.4f}")
        
        return aggregated_params, modality_summary

    def aggregate_within_dataset_modality(self, 
                                          signature: str, 
                                          client_results: List[Tuple[ClientProxy, FitRes]]) -> Tuple[NDArrays, Dict]:
        """Aggregate parameters within a specific dataset-modality group (enhanced for multi-dataset)"""
        print(f"[Server] Aggregating {len(client_results)} clients within dataset-modality group: {signature}")
        
        # Extract parameters and weights for this dataset-modality group
        group_params_and_weights = []
        total_examples = 0
        group_metrics = []
        
        for client_proxy, fit_res in client_results:
            client_id = str(fit_res.metrics.get("client_id", "unknown"))
            client_signature = self.client_signatures.get(client_id, "UNKNOWN_UNKNOWN")
            
            if client_signature == signature:
                parameters = parameters_to_ndarrays(fit_res.parameters)
                num_examples = fit_res.num_examples
                group_params_and_weights.append((parameters, num_examples))
                total_examples += num_examples
                group_metrics.append(fit_res.metrics)
        
        if not group_params_and_weights:
            return None, {}
        
        # Use asymmetric aggregation if we have the client results 
        if len(group_params_and_weights) > 1:
            # Extract client info for asymmetric aggregation
            group_client_results = [(client_proxy, fit_res) for client_proxy, fit_res in client_results 
                                   if self.client_signatures.get(str(fit_res.metrics.get("client_id", "unknown")), "UNKNOWN_UNKNOWN") == signature]
            
            if len(group_client_results) > 1:
                # Use simple common layer aggregation for dataset-modality groups
                client_info = self._extract_client_parameter_info(group_client_results)
                compatible_params = self._find_common_layers(client_info)
                
                if compatible_params and len(compatible_params) > 0:
                    print(f"[Server] Dataset-modality {signature}: aggregating {len(compatible_params)} common parameters")
                    aggregated_param_dict = self._asymmetric_weighted_average(client_info, compatible_params)
                    common_param_names = sorted(compatible_params.keys())
                    aggregated_params = [aggregated_param_dict[name] for name in common_param_names]
                    self.last_common_param_names = common_param_names
                else:
                    print(f"[Server] Dataset-modality {signature}: no common parameters found")
                    return None, {}
            else:
                aggregated_params = self._weighted_average(group_params_and_weights)
        else:
            aggregated_params = self._weighted_average(group_params_and_weights)
        
        # Calculate group-specific metrics
        group_validation_scores = []
        for metrics in group_metrics:
            validation_dice = metrics.get("validation_dice", {})
            if isinstance(validation_dice, dict) and "mean" in validation_dice:
                group_validation_scores.append(validation_dice["mean"])
        
        avg_group_dice = np.mean(group_validation_scores) if group_validation_scores else 0.0
        
        # Parse signature for display
        dataset_part, modality_part = signature.split('_', 1) if '_' in signature else (signature, 'UNKNOWN')
        
        group_summary = {
            'signature': signature,
            'dataset': dataset_part,
            'modality': modality_part,
            'num_clients': len(group_params_and_weights),
            'total_examples': total_examples,
            'avg_validation_dice': avg_group_dice,
            'validation_scores': group_validation_scores
        }
        
        print(f"[Server] Dataset-Modality {signature}: {len(group_params_and_weights)} clients, "
              f"{total_examples} examples, avg_dice={avg_group_dice:.4f}")
        
        return aggregated_params, group_summary

    def aggregate_across_modalities(self, 
                                    modality_results: Dict[str, Tuple[NDArrays, Dict]]) -> Tuple[NDArrays, Dict]:
        """Aggregate the per-modality models into a global model"""
        print(f"[Server] Aggregating across {len(modality_results)} modalities")
        
        # Prepare weighted aggregation across modalities
        inter_modality_params_and_weights = []
        total_weight = 0.0
        aggregation_summary = {}
        
        for modality, (params, summary) in modality_results.items():
            if params is None:
                continue
                
            # Determine weight for this modality
            if self.modality_weights and modality in self.modality_weights:
                # Use specified weight
                modality_weight = self.modality_weights[modality]
            else:
                # Use number of examples as weight
                modality_weight = summary['total_examples']
            
            inter_modality_params_and_weights.append((params, modality_weight))
            total_weight += modality_weight
            
            aggregation_summary[modality] = {
                'weight': modality_weight,
                'num_clients': summary['num_clients'],
                'total_examples': summary['total_examples'],
                'avg_validation_dice': summary['avg_validation_dice']
            }
            
            print(f"[Server] Modality {modality}: weight={modality_weight:.2f}, "
                  f"dice={summary['avg_validation_dice']:.4f}")
        
        if not inter_modality_params_and_weights:
            return None, {}
        
        # Perform weighted averaging across modalities
        global_params = self._weighted_average(inter_modality_params_and_weights)
        
        # Calculate global metrics
        global_avg_dice = sum(
            summary['avg_validation_dice'] * summary['weight'] 
            for summary in aggregation_summary.values()
        ) / total_weight if total_weight > 0 else 0.0
        
        global_summary = {
            'global_avg_validation_dice': global_avg_dice,
            'total_weight': total_weight,
            'modality_breakdown': aggregation_summary,
            'aggregation_method': 'modality_aware'
        }
        
        print(f"[Server] Global aggregation: dice={global_avg_dice:.4f}, total_weight={total_weight:.2f}")
        
        return global_params, global_summary

    def aggregate_multi_dataset_aware(self, 
                                      client_results: List[Tuple[ClientProxy, FitRes]]) -> Tuple[NDArrays, Dict]:
        """
        Enhanced aggregation that handles multi-dataset federation.
        First aggregates within dataset-modality groups, then across groups.
        """
        print(f"[Server] Performing multi-dataset aware aggregation...")
        
        # Group aggregation within dataset-modality signatures
        dataset_modality_results = {}
        
        for signature in self.dataset_modality_groups.keys():
            if len(self.dataset_modality_groups[signature]) > 0:
                group_params, group_summary = self.aggregate_within_dataset_modality(signature, client_results)
                if group_params is not None:
                    dataset_modality_results[signature] = (group_params, group_summary)
        
        if not dataset_modality_results:
            print("[Server] No valid dataset-modality groups found")
            return None, {}
        
        print(f"[Server] Aggregating across {len(dataset_modality_results)} dataset-modality groups")
        
        # Aggregate across dataset-modality groups
        inter_group_params_and_weights = []
        total_weight = 0.0
        aggregation_summary = {}
        
        for signature, (params, summary) in dataset_modality_results.items():
            # Determine weight for this dataset-modality group
            dataset = summary['dataset']
            modality = summary['modality']
            
            # Check for specific dataset-modality weights first
            weight_key = f"{dataset}_{modality}"
            if self.modality_weights and weight_key in self.modality_weights:
                group_weight = self.modality_weights[weight_key]
            elif self.modality_weights and modality in self.modality_weights:
                # Fall back to modality-only weights
                group_weight = self.modality_weights[modality]
            else:
                # Use number of examples as weight
                group_weight = summary['total_examples']
            
            inter_group_params_and_weights.append((params, group_weight))
            total_weight += group_weight
            
            aggregation_summary[signature] = {
                'weight': group_weight,
                'dataset': dataset,
                'modality': modality,
                'num_clients': summary['num_clients'],
                'total_examples': summary['total_examples'],
                'avg_validation_dice': summary['avg_validation_dice']
            }
            
            print(f"[Server] Group {signature}: weight={group_weight:.2f}, "
                  f"dice={summary['avg_validation_dice']:.4f}")
        
        # Perform weighted averaging across dataset-modality groups
        global_params = self._weighted_average(inter_group_params_and_weights)
        
        # Calculate global metrics
        global_avg_dice = sum(
            summary['avg_validation_dice'] * summary['weight'] 
            for summary in aggregation_summary.values()
        ) / total_weight if total_weight > 0 else 0.0
        
        global_summary = {
            'global_avg_validation_dice': global_avg_dice,
            'total_weight': total_weight,
            'dataset_modality_breakdown': aggregation_summary,
            'aggregation_method': 'multi_dataset_modality_aware',
            'num_dataset_modality_groups': len(dataset_modality_results),
            'unique_datasets': len(set(s['dataset'] for s in aggregation_summary.values())),
            'unique_modalities': len(set(s['modality'] for s in aggregation_summary.values()))
        }
        
        print(f"[Server] Multi-dataset aggregation complete:")
        print(f"  Global Dice: {global_avg_dice:.4f}")
        print(f"  Total weight: {total_weight:.2f}")
        print(f"  Groups: {len(dataset_modality_results)}")
        print(f"  Datasets: {global_summary['unique_datasets']}")
        print(f"  Modalities: {global_summary['unique_modalities']}")
        
        return global_params, global_summary

    def _extract_client_parameter_info(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict:
        """
        Extract parameter metadata from client metrics following FednnUNet approach.
        Parameter shapes are derived from received parameters to ensure consistency.
        Returns: {client_id: {"param_names": [], "param_shapes": [], "parameters": [], "weight": float, "architecture_signature": str}}
        """
        client_info = {}
        
        for client_proxy, fit_res in results:
            client_id = str(fit_res.metrics.get("client_id", "unknown"))
            
            # Extract parameter metadata from client metrics
            param_names_str = fit_res.metrics.get("param_names_str", "[]")
            try:
                param_names = json.loads(param_names_str) if param_names_str else []
            except json.JSONDecodeError:
                param_names = fit_res.metrics.get("param_names", [])  # Fallback
            architecture_signature = fit_res.metrics.get("architecture_signature", "unknown")

            # Convert Flower parameters to NDArrays and derive shapes
            parameters = parameters_to_ndarrays(fit_res.parameters)
            param_shapes = [list(p.shape) for p in parameters]

            client_info[client_id] = {
                "param_names": param_names,
                "param_shapes": param_shapes,
                "parameters": parameters,
                "weight": float(fit_res.num_examples),  # Use number of examples as weight
                "num_examples": fit_res.num_examples,
                "architecture_signature": architecture_signature,
                "metrics": fit_res.metrics
            }
            
            print(f"[Server] Client {client_id}: extracted {len(param_names)} parameter names (arch: {architecture_signature})")
        
        print(f"[Server] Successfully extracted parameter info from {len(client_info)} clients")
        return client_info

    def _find_common_layers(self, client_info: Dict) -> Dict:
        """
        Enhanced parameter intersection with strict (name, shape) matching.
        Only aggregates parameters present in ALL clients with IDENTICAL shapes.
        Returns: {param_name: {"shape": shape, "clients": [client_ids], "client_count": int, "is_batchnorm_learnable": bool}}
        """
        if not client_info:
            return {}
        
        print(f"[Server] Finding common parameters across {len(client_info)} clients using enhanced intersection")
        
        # Log detailed client information
        total_client_params = 0
        for client_id, info in client_info.items():
            arch_sig = info.get("architecture_signature", "unknown")
            num_params = len(info["param_names"])
            weight = info.get("weight", "unknown")
            total_client_params += num_params
            print(f"[Server] Client {client_id}: {num_params} parameters, weight: {weight}, architecture: {arch_sig}")
        
        print(f"[Server] Total parameters across all clients: {total_client_params}")
        
        # Create (name, shape) tuples for intersection
        client_ids = list(client_info.keys())
        if not client_ids:
            return {}
        
        # Build parameter signature sets for each client: {(name, shape_tuple): client_id}
        client_param_signatures = {}
        for client_id, info in client_info.items():
            signatures = set()
            for i, param_name in enumerate(info["param_names"]):
                param_shape = tuple(info["param_shapes"][i])  # Ensure tuple for hashing
                signatures.add((param_name, param_shape))
            client_param_signatures[client_id] = signatures
            print(f"[Server] Client {client_id}: {len(signatures)} unique (name, shape) signatures")
        
        # Find intersection of (name, shape) signatures across ALL clients
        common_signatures = client_param_signatures[client_ids[0]]
        for client_id in client_ids[1:]:
            common_signatures = common_signatures.intersection(client_param_signatures[client_id])
        
        print(f"[Server] Found {len(common_signatures)} common (name, shape) signatures across all clients")
        
        # Build compatible parameters dictionary
        compatible_params = {}
        batchnorm_learnable_count = 0
        batchnorm_running_count = 0
        
        for param_name, param_shape in common_signatures:
            # Determine if this is a BatchNorm learnable parameter
            is_batchnorm_learnable = self._is_batchnorm_learnable_param(param_name)
            is_batchnorm_running = self._is_batchnorm_running_stat(param_name)
            
            if is_batchnorm_learnable:
                batchnorm_learnable_count += 1
            elif is_batchnorm_running:
                batchnorm_running_count += 1
                # Skip BatchNorm running statistics by default
                print(f"[Server] Excluding BatchNorm running stat: {param_name} (shape: {param_shape})")
                continue
            
            compatible_params[param_name] = {
                "shape": param_shape,
                "clients": client_ids.copy(),  # All clients have this parameter
                "client_count": len(client_ids),
                "is_batchnorm_learnable": is_batchnorm_learnable
            }
        
        # Log detailed statistics
        print(f"[Server] ===== PARAMETER INTERSECTION SUMMARY =====")
        print(f"[Server] Total compatible parameters: {len(compatible_params)}")
        print(f"[Server] BatchNorm learnable parameters: {batchnorm_learnable_count}")
        print(f"[Server] BatchNorm running stats excluded: {batchnorm_running_count}")
        print(f"[Server] Regular backbone parameters: {len(compatible_params) - batchnorm_learnable_count}")
        
        # Log sample parameters for verification
        sample_params = list(compatible_params.keys())[:5]
        print(f"[Server] Sample compatible parameters:")
        for param_name in sample_params:
            shape = compatible_params[param_name]["shape"]
            is_bn = compatible_params[param_name]["is_batchnorm_learnable"]
            bn_flag = " (BatchNorm)" if is_bn else ""
            print(f"[Server]   {param_name} -> {shape}{bn_flag}")
        
        if len(compatible_params) == 0:
            print(f"[Server] WARNING: No compatible parameters found for aggregation!")
            print(f"[Server] This may indicate architecture incompatibility between clients.")
        
        return compatible_params

    def _is_batchnorm_learnable_param(self, param_name: str) -> bool:
        """
        Check if a parameter is a BatchNorm learnable parameter (weight or bias).
        Returns True for gamma (weight) and beta (bias) parameters.
        """
        # Remove wrapper prefixes
        clean_name = param_name.replace("_orig_mod.", "").replace("network.", "").replace("model.", "")
        
        # Check for norm layer indicators and learnable parameter types
        norm_indicators = [".norm.", ".bn.", ".batch_norm.", "_norm.", "_bn."]
        learnable_types = [".weight", ".bias"]
        
        has_norm = any(indicator in clean_name for indicator in norm_indicators)
        has_learnable = any(param_type in clean_name for param_type in learnable_types)
        
        return has_norm and has_learnable
    
    def _is_batchnorm_running_stat(self, param_name: str) -> bool:
        """
        Check if a parameter is a BatchNorm running statistic.
        Returns True for running_mean, running_var, num_batches_tracked.
        """
        # Remove wrapper prefixes
        clean_name = param_name.replace("_orig_mod.", "").replace("network.", "").replace("model.", "")
        
        # Check for running statistics
        running_stats = ["running_mean", "running_var", "num_batches_tracked"]
        norm_indicators = [".norm.", ".bn.", ".batch_norm.", "_norm.", "_bn."]
        
        has_norm = any(indicator in clean_name for indicator in norm_indicators)
        has_running_stat = any(stat in clean_name for stat in running_stats)
        
        return has_norm and has_running_stat

    def _find_common_semantic_parameters(self, client_info: Dict) -> Dict:
        """
        Find semantic parameters that exist in multiple clients with identical shapes.
        Enhanced implementation for semantic parameter matching across heterogeneous architectures.
        Returns: {semantic_param_name: {"shape": shape, "clients": [client_ids], "client_count": int, "architecture_signatures": [sigs]}}
        """
        if not client_info:
            return {}
        
        print(f"[Server] Finding common semantic parameters across {len(client_info)} clients")
        
        # Log client architecture signatures
        architectures = {}
        for client_id, info in client_info.items():
            arch_sig = info.get("architecture_signature", "unknown")
            if arch_sig not in architectures:
                architectures[arch_sig] = []
            architectures[arch_sig].append(client_id)
            
        print(f"[Server] Client architectures found: {dict(architectures)}")
        
        # Collect all unique semantic parameter names across clients
        all_semantic_names = set()
        for info in client_info.values():
            all_semantic_names.update(info["semantic_param_names"])
        
        compatible_params = {}
        
        # Check each semantic parameter for compatibility across clients
        for semantic_name in all_semantic_names:
            clients_with_param = []
            param_shapes = []
            client_architectures = []
            
            # Find which clients have this semantic parameter and their shapes
            for client_id, info in client_info.items():
                if semantic_name in info["semantic_param_names"]:
                    param_idx = info["semantic_param_names"].index(semantic_name)
                    param_shape = info["param_shapes"][param_idx]
                    
                    clients_with_param.append(client_id)
                    param_shapes.append(param_shape)
                    client_architectures.append(info.get("architecture_signature", "unknown"))
            
            # Only include if multiple clients have it with identical shapes
            if len(clients_with_param) > 1:
                # Check if all shapes are identical
                if all(tuple(shape) == tuple(param_shapes[0]) for shape in param_shapes):
                    compatible_params[semantic_name] = {
                        "shape": tuple(param_shapes[0]),
                        "clients": clients_with_param,
                        "client_count": len(clients_with_param),
                        "architecture_signatures": client_architectures
                    }
                    print(f"[Server] Common semantic parameter: {semantic_name} -> {param_shapes[0]} (clients: {clients_with_param})")
                else:
                    print(f"[Server] Shape mismatch for semantic parameter {semantic_name}: {param_shapes} (clients: {clients_with_param})")
            else:
                print(f"[Server] Semantic parameter {semantic_name} only exists in {len(clients_with_param)} client(s), skipping")
        
        print(f"[Server] Found {len(compatible_params)} common semantic parameters out of {len(all_semantic_names)} total unique semantic parameters")
        
        # Enhanced compatibility analysis
        total_possible_pairs = len(client_info) * (len(client_info) - 1) // 2
        if len(compatible_params) < len(all_semantic_names) * 0.3:
            print(f"[Server] LOW COMPATIBILITY: Only {len(compatible_params)}/{len(all_semantic_names)} semantic parameters are common")
            print(f"[Server] This suggests very heterogeneous architectures - expect limited aggregation effectiveness")
        elif len(compatible_params) < len(all_semantic_names) * 0.7:
            print(f"[Server] MODERATE COMPATIBILITY: {len(compatible_params)}/{len(all_semantic_names)} semantic parameters are common")
            print(f"[Server] Reasonably compatible architectures - aggregation should be effective")
        else:
            print(f"[Server] HIGH COMPATIBILITY: {len(compatible_params)}/{len(all_semantic_names)} semantic parameters are common")
            print(f"[Server] Very compatible architectures - optimal aggregation expected")
        
        return compatible_params

    def _find_compatible_parameters(self, client_info: Dict) -> Dict:
        """
        Simple method for finding compatible parameters - uses intersection approach.
        """
        return self._find_common_layers(client_info)

    def _semantic_weighted_average(self, client_info: Dict, compatible_params: Dict) -> Dict:
        """
        Enhanced asymmetric aggregation using semantic parameter names.
        Only aggregates semantic parameters that exist in multiple clients with same shape.
        Returns: {semantic_param_name: aggregated_parameter}
        """
        if not compatible_params:
            print(f"[Server] No common semantic parameters found for aggregation")
            return {}
        
        print(f"[Server] Performing semantic weighted average on {len(compatible_params)} common parameters")
        
        aggregated_params = {}
        aggregation_stats = {
            "total_aggregated": 0,
            "total_skipped": 0,
            "architecture_cross_aggregation": 0,
            "same_architecture_aggregation": 0
        }
        
        for semantic_name, compat_info in compatible_params.items():
            # Collect parameter values and weights from compatible clients
            param_values = []
            weights = []
            client_architectures = []
            
            for client_id in compat_info["clients"]:
                client_data = client_info[client_id]
                param_idx = client_data["semantic_param_names"].index(semantic_name)
                param_value = client_data["parameters"][param_idx]
                client_weight = client_data["weight"]
                client_arch = client_data.get("architecture_signature", "unknown")
                
                param_values.append(param_value)
                weights.append(client_weight)
                client_architectures.append(client_arch)
            
            # Check if this is cross-architecture aggregation
            unique_architectures = set(client_architectures)
            if len(unique_architectures) > 1:
                aggregation_stats["architecture_cross_aggregation"] += 1
                print(f"[Server] Cross-architecture aggregation for {semantic_name}: {list(unique_architectures)}")
            else:
                aggregation_stats["same_architecture_aggregation"] += 1
            
            # Perform weighted average: enhanced version of θ^l = 1/|𝒦l| * ∑(k∈𝒦l) θlk
            total_weight = sum(weights)
            if total_weight == 0:
                total_weight = len(weights)  # Equal weighting fallback
            
            # Weighted aggregation
            aggregated_param = None
            for param_value, weight in zip(param_values, weights):
                contribution = (weight / total_weight) * param_value
                if aggregated_param is None:
                    aggregated_param = contribution
                else:
                    aggregated_param += contribution
            
            aggregated_params[semantic_name] = aggregated_param
            aggregation_stats["total_aggregated"] += 1
            
            print(f"[Server] Aggregated semantic parameter {semantic_name} from {len(param_values)} clients (shape: {compat_info['shape']})")
        
        print(f"[Server] Semantic aggregation completed:")
        print(f"[Server]   Aggregated parameters: {aggregation_stats['total_aggregated']}")
        print(f"[Server]   Cross-architecture aggregations: {aggregation_stats['architecture_cross_aggregation']}")
        print(f"[Server]   Same-architecture aggregations: {aggregation_stats['same_architecture_aggregation']}")
        
        return aggregated_params

    def _asymmetric_weighted_average(self, client_info: Dict, compatible_params: Dict) -> Dict:
        """
        Enhanced weighted FedAvg aggregation with data size weighting.
        Only aggregates parameters that exist in ALL clients with identical shapes.
        Uses data-proportional weighting and proper weight validation.
        Returns: {param_name: aggregated_parameter}
        """
        if not compatible_params:
            print(f"[Server] No compatible parameters found for aggregation")
            return {}
        
        print(f"[Server] Performing enhanced weighted FedAvg on {len(compatible_params)} compatible parameters")
        
        # Validate and normalize client weights
        total_clients = len(client_info)
        client_weights_info = {}
        total_weight = 0
        
        for client_id, info in client_info.items():
            weight = info.get("weight", 1.0)  # Default weight if missing
            total_weight += weight
            client_weights_info[client_id] = weight
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for client_id in client_weights_info:
                client_weights_info[client_id] /= total_weight
        else:
            # Equal weighting fallback
            equal_weight = 1.0 / total_clients
            for client_id in client_weights_info:
                client_weights_info[client_id] = equal_weight
        
        # Log weight distribution
        print(f"[Server] ===== CLIENT WEIGHT DISTRIBUTION =====")
        for client_id, normalized_weight in client_weights_info.items():
            original_weight = client_info[client_id].get("weight", 1.0)
            print(f"[Server] Client {client_id}: original_weight={original_weight:.2f}, normalized_weight={normalized_weight:.4f}")
        
        # Validate weights sum to 1.0
        weight_sum = sum(client_weights_info.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights do not sum to 1.0: {weight_sum}"
        
        aggregated_params = {}
        batchnorm_aggregated = 0
        regular_aggregated = 0
        
        for param_name, compat_info in compatible_params.items():
            # Collect parameter values and normalized weights
            param_values = []
            normalized_weights = []
            
            for client_id in compat_info["clients"]:
                client_data = client_info[client_id]
                param_idx = client_data["param_names"].index(param_name)
                param_value = client_data["parameters"][param_idx]
                normalized_weight = client_weights_info[client_id]
                
                param_values.append(param_value)
                normalized_weights.append(normalized_weight)
            
            # Verify weights for this parameter sum to approximately 1.0
            param_weight_sum = sum(normalized_weights)
            if abs(param_weight_sum - 1.0) > 1e-6:
                print(f"[Server] WARNING: Weights for {param_name} sum to {param_weight_sum:.6f}, not 1.0")
            
            # Perform weighted FedAvg: θ_global = Σ(w_i * θ_i)
            aggregated_param = None
            for param_value, weight in zip(param_values, normalized_weights):
                weighted_contribution = weight * param_value
                if aggregated_param is None:
                    aggregated_param = weighted_contribution.clone() if hasattr(weighted_contribution, 'clone') else weighted_contribution
                else:
                    aggregated_param += weighted_contribution
            
            aggregated_params[param_name] = aggregated_param
            
            # Count parameter types
            if compat_info.get("is_batchnorm_learnable", False):
                batchnorm_aggregated += 1
            else:
                regular_aggregated += 1
        
        # Log aggregation summary
        print(f"[Server] ===== FEDAVG AGGREGATION SUMMARY =====")
        print(f"[Server] Total parameters aggregated: {len(aggregated_params)}")
        print(f"[Server] BatchNorm learnable parameters: {batchnorm_aggregated}")
        print(f"[Server] Regular backbone parameters: {regular_aggregated}")
        print(f"[Server] Clients participating: {total_clients}")
        
        # Log sample aggregated parameters
        sample_params = list(aggregated_params.keys())[:3]
        for param_name in sample_params:
            shape = compatible_params[param_name]["shape"]
            is_bn = compatible_params[param_name].get("is_batchnorm_learnable", False)
            bn_flag = " (BatchNorm)" if is_bn else ""
            print(f"[Server] Aggregated: {param_name} -> {shape}{bn_flag}")

        # Update last_common_param_names with aggregated parameter names
        existing = set(getattr(self, "last_common_param_names", []))
        self.last_common_param_names = list(existing.union(aggregated_params.keys()))

        return aggregated_params

    def _intelligent_fallback_aggregation(self, client_info: Dict, compatible_params: Dict) -> Tuple[NDArrays, Dict]:
        """
        Intelligent fallback strategy when insufficient common parameters are found.
        Implements multiple fallback approaches based on the compatibility level.
        """
        if not client_info:
            print("[Server] ERROR: No client info available for fallback aggregation")
            return [], {"aggregation_method": "error_no_clients"}
        
        # Calculate compatibility ratio
        all_param_names = [info["param_names"] for info in client_info.values()]
        total_unique_params = len(set().union(*all_param_names)) if all_param_names else 0
        common_param_ratio = len(compatible_params) / max(total_unique_params, 1)
        
        print(f"[Server] Insufficient common parameters ({len(compatible_params)}/{total_unique_params}, ratio: {common_param_ratio:.2f})")
        print(f"[Server] Applying intelligent fallback aggregation strategy...")
        
        if common_param_ratio >= 0.1 and compatible_params:  # At least 10% common parameters
            print("[Server] PARTIAL AGGREGATION: Aggregating available common parameters")
            
            # Aggregate what we can
            aggregated_params = self._asymmetric_weighted_average(client_info, compatible_params)
            
            # Select the most representative client as the base
            best_client_id = max(client_info.keys(), key=lambda cid: client_info[cid]["weight"])
            best_client_info = client_info[best_client_id]
            
            print(f"[Server] Using client {best_client_id} as base (weight: {best_client_info['weight']})")
            
            # Create a hybrid parameter set: aggregated common + best client's unique
            final_params = []
            
            for i, param_name in enumerate(best_client_info["param_names"]):
                if param_name in aggregated_params:
                    final_params.append(aggregated_params[param_name])
                else:
                    final_params.append(best_client_info["parameters"][i])
            
            summary = {
                "aggregation_method": "partial_aggregation",
                "common_parameters_aggregated": len(aggregated_params),
                "unique_parameters_from_best": len(final_params) - len(aggregated_params),
                "fallback_client": best_client_id,
                "compatibility_ratio": common_param_ratio
            }
            
            return final_params, summary
            
        # Very low compatibility - use weighted client selection
        print("[Server] CLIENT SELECTION FALLBACK: Using best-performing client's parameters")
        
        # Select client based on performance or weight
        best_client_id = max(client_info.keys(), key=lambda cid: client_info[cid]["weight"])
        best_client_info = client_info[best_client_id]
        
        print(f"[Server] Selected client {best_client_id} (weight: {best_client_info['weight']}, arch: {best_client_info.get('architecture_signature', 'unknown')})")
        
        summary = {
            "aggregation_method": "client_selection_fallback",
            "selected_client": best_client_id,
            "selected_client_weight": best_client_info["weight"],
            "selected_client_arch": best_client_info.get("architecture_signature", "unknown"),
            "compatibility_ratio": common_param_ratio,
            "total_parameters": len(best_client_info["parameters"])
        }
        
        return best_client_info["parameters"], summary

    def _distribute_parameters_to_clients(self, aggregated_params: Dict, client_info: Dict) -> Dict:
        """
        Return aggregated parameters to each client based on their architecture.
        Each client gets back only the parameters they can use.
        Returns: {client_id: [param_list_in_client_order]}
        """
        client_distributions = {}
        
        for client_id, client_data in client_info.items():
            client_params = []
            
            # For each parameter this client expects (in their original order)
            for param_name in client_data["param_names"]:
                if param_name in aggregated_params:
                    # Give them the aggregated version
                    client_params.append(aggregated_params[param_name])
                else:
                    # Keep their original parameter (not aggregated)
                    param_idx = client_data["param_names"].index(param_name)
                    client_params.append(client_data["parameters"][param_idx])
            
            client_distributions[client_id] = client_params
            print(f"[Server] Client {client_id}: prepared {len(client_params)} parameters ({len([p for p in client_data['param_names'] if p in aggregated_params])} aggregated, {len(client_params) - len([p for p in client_data['param_names'] if p in aggregated_params])} original)")
        
        return client_distributions

    def _weighted_average(self, params_and_weights: List[Tuple[NDArrays, float]]) -> NDArrays:
        """Perform simple weighted average of backbone parameters"""
        if not params_and_weights:
            return []
        
        return self._backbone_weighted_average(params_and_weights)
    
    def _backbone_weighted_average(self, params_and_weights: List[Tuple[NDArrays, float]]) -> NDArrays:
        """
        Safe fallback aggregation that avoids shape mismatch errors.
        Returns parameters from the client with highest weight when shapes don't match.
        """
        print(f"[Server] Performing backbone aggregation for {len(params_and_weights)} clients")
        
        if not params_and_weights:
            return []
        
        # If only one client, return their parameters
        if len(params_and_weights) == 1:
            params, _ = params_and_weights[0]
            print(f"[Server] Single client aggregation - returning {len(params)} parameters")
            return params
        
        print(f"[Server] Multi-client aggregation - checking parameter compatibility")
        
        # Check if all clients have the same number of parameters
        param_counts = [len(params) for params, _ in params_and_weights]
        if len(set(param_counts)) > 1:
            print(f"[Server] WARNING: Clients have different parameter counts: {param_counts}")
            print(f"[Server] This indicates incompatible architectures - using highest weight client")
            
            # Return parameters from client with highest weight
            best_client_idx = max(range(len(params_and_weights)), key=lambda i: params_and_weights[i][1])
            best_params, best_weight = params_and_weights[best_client_idx]
            print(f"[Server] Using parameters from client with weight {best_weight} ({len(best_params)} parameters)")
            return best_params
        
        # All clients have same number of parameters - check overall compatibility first
        first_params, first_weight = params_and_weights[0]
        
        # Pre-check: verify if architectures are actually compatible by comparing all shapes at once
        all_shapes_compatible = True
        incompatible_layers = []
        
        for i in range(len(first_params)):
            shapes = []
            for params, _ in params_and_weights:
                if i < len(params) and hasattr(params[i], 'shape'):
                    shapes.append(params[i].shape)
            
            if len(set(tuple(shape) for shape in shapes)) > 1:
                all_shapes_compatible = False
                incompatible_layers.append((i, shapes))
        
        # If we have many incompatible layers, the architectures are fundamentally different
        incompatible_ratio = len(incompatible_layers) / len(first_params)
        
        if not all_shapes_compatible and incompatible_ratio > 0.3:  # More than 30% incompatible
            print(f"[Server] WARNING: {incompatible_ratio:.1%} of layers have incompatible shapes")
            print(f"[Server] Architectures are fundamentally different - using highest weight client")
            print(f"[Server] Sample incompatible shapes: {incompatible_layers[:3]}")  # Show first 3 examples
            
            # Return parameters from client with highest weight
            best_client_idx = max(range(len(params_and_weights)), key=lambda i: params_and_weights[i][1])
            best_params, best_weight = params_and_weights[best_client_idx]
            print(f"[Server] Using parameters from client with weight {best_weight}")
            return best_params
        
        # Proceed with layer-by-layer aggregation for minor incompatibilities
        print(f"[Server] Performing layer-by-layer aggregation ({len(incompatible_layers)} incompatible layers)")
        aggregated_params = []
        
        for i in range(len(first_params)):
            # Check if all parameters at this index have compatible shapes
            shapes = []
            param_values = []
            weights = []
            
            for params, weight in params_and_weights:
                if i < len(params) and hasattr(params[i], 'shape'):
                    shapes.append(params[i].shape)
                    param_values.append(params[i])
                    weights.append(weight)
            
            # Only aggregate if all shapes are identical
            unique_shapes = set(tuple(shape) for shape in shapes)
            if len(unique_shapes) == 1:
                # Safe to aggregate - all shapes match
                total_weight = sum(weights)
                if total_weight == 0:
                    total_weight = len(weights)
                
                weighted_sum = None
                for param_value, weight in zip(param_values, weights):
                    contribution = (weight / total_weight) * param_value
                    if weighted_sum is None:
                        weighted_sum = contribution
                    else:
                        weighted_sum += contribution
                
                aggregated_params.append(weighted_sum)
            else:
                # Shape mismatch - use parameter from client with highest weight
                print(f"[Server] Shape mismatch at layer {i}: {shapes} - using highest weight client")
                best_idx = max(range(len(weights)), key=lambda idx: weights[idx])
                aggregated_params.append(param_values[best_idx])
        
        print(f"[Server] Safe aggregation completed - aggregated {len(aggregated_params)} parameters")
        return aggregated_params

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the fit round with multi-phase nnUNet federated learning."""
        
        # Determine federated round based on server round
        if server_round == 1:
            federated_round = -2  # Preprocessing round
            self.current_phase = "preprocessing"
        elif server_round == 2:
            federated_round = -1  # Initialization round  
            self.current_phase = "initialization"
        else:
            federated_round = server_round - 2  # Training rounds (0, 1, 2, ...)
            self.current_phase = "training"
            
        print(f"[Server] Round {server_round} -> Federated round {federated_round} ({self.current_phase} phase)")
        
        config = {
            "server_round": federated_round,
            "local_epochs": int(os.environ.get('LOCAL_EPOCHS', 2)),
            "validate": federated_round >= 0,  # Enable validation for training rounds
            "enable_modality_metadata": True,  # Request modality information from clients
            "exclude_incompatible_layers": True,  # Enable architecture-aware parameter filtering
            "backbone_aggregation": True,  # Enable backbone-only aggregation
        }

        if hasattr(self, "last_common_param_names"):
            config["param_names_str"] = json.dumps(self.last_common_param_names)
        
        # Call parent's configure_fit method
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        
        # Update config in fit instructions
        updated_fit_ins = []
        for client_proxy, fit_ins_item in fit_ins:
            fit_ins_item.config.update(config)
            updated_fit_ins.append((client_proxy, fit_ins_item))
            
        return updated_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        print(f"[Server] Round {server_round} results: {len(results)} successes, {len(failures)} failures.")

        # Reset tracking of common parameter names for this round
        self.last_common_param_names = []
        
        # Determine federated round
        if server_round == 1:
            federated_round = -2  # Preprocessing round
        elif server_round == 2:
            federated_round = -1  # Initialization round
        else:
            federated_round = server_round - 2  # Training rounds

        # Handle preprocessing round - collect fingerprints and modality info
        if federated_round == -2:
            print("[Server] Processing fingerprint collection round")
            for _, fitres in results:
                client_id = str(fitres.metrics.get("client_id", "unknown"))
                self.update_client_modality_mapping(client_id, fitres.metrics)
                
                preprocessing_complete = fitres.metrics.get("preprocessing_complete", False)
                fingerprint_cases = fitres.metrics.get("fingerprint_cases", 0)
                if preprocessing_complete:
                    print(f"[Server] Client {client_id} preprocessed {fingerprint_cases} cases")
                    
            print(f"[Server] Collected {len(results)}/{self.expected_num_clients} fingerprints")
            print(f"[Server] Detected modality groups: {dict(self.modality_groups)}")
            print(f"[Server] Detected dataset-modality groups: {dict(self.dataset_modality_groups)}")
            
            return None, {}

        # Handle initialization round
        elif federated_round == -1:
            print("[Server] Processing initialization round")
            for _, fitres in results:
                client_id = str(fitres.metrics.get("client_id", "unknown"))
                init_complete = fitres.metrics.get("initialization_complete", False)
                if init_complete:
                    print(f"[Server] Client {client_id} initialized successfully")
                    
            return None, {}

        # Handle regular training rounds with modality-aware aggregation
        else:
            print(f"[Server] Processing training round {federated_round}")
            
            # Update modality mappings and collect metrics, handle warmup status
            client_validation_scores = []
            warmup_clients = []
            for _, fitres in results:
                client_id = str(fitres.metrics.get("client_id", "unknown"))
                self.update_client_modality_mapping(client_id, fitres.metrics)
                
                # Track warmup status
                is_warmup = fitres.metrics.get("is_warmup", False)
                warmup_complete = fitres.metrics.get("warmup_complete", False)
                
                if is_warmup:
                    warmup_clients.append(client_id)
                    if warmup_complete:
                        self.client_warmup_status[client_id] = True
                        print(f"[Server] Client {client_id}: completed warmup phase")
                    else:
                        print(f"[Server] Client {client_id}: in warmup phase")
                
                loss = fitres.metrics.get("loss", 0.0)
                epochs = fitres.metrics.get("local_epochs_completed", 0)
                validation_dice = fitres.metrics.get("validation_dice", {})
                
                if isinstance(validation_dice, dict) and "mean" in validation_dice:
                    val_score = validation_dice["mean"]
                    client_validation_scores.append(val_score)
                    print(f"[Server] Client {client_id}: loss={loss:.4f}, epochs={epochs}, val_dice={val_score:.4f}")
                else:
                    print(f"[Server] Client {client_id}: loss={loss:.4f}, epochs={epochs}")
            
            # Special handling for round 0 (warmup round)
            if federated_round == 0:
                print(f"[Server] Round 0 warmup: {len(warmup_clients)} clients warming up")
                # For warmup round, just collect parameters but don't aggregate globally yet
                # Each client is training their first/last layers independently
                if len(warmup_clients) > 0:
                    print(f"[Server] Warmup clients: {warmup_clients}")
                    # Return the first client's parameters as a placeholder for now
                    if results:
                        return (results[0][1].parameters, {"warmup_round": True, "warmup_clients": len(warmup_clients)})
                    else:
                        return None, {}
                else:
                    print("[Server] No warmup clients in round 0")
                    return None, {}
            
            # Determine aggregation strategy
            num_unique_datasets = len(set(self.client_datasets.values()))
            num_unique_modalities = len(self.modality_groups)
            num_dataset_modality_groups = len(self.dataset_modality_groups)
            
            print(f"[Server] Aggregation analysis:")
            print(f"  Unique datasets: {num_unique_datasets}")
            print(f"  Unique modalities: {num_unique_modalities}")
            print(f"  Dataset-modality groups: {num_dataset_modality_groups}")
            
            # Choose aggregation strategy
            print(f"[Server] Strategy decision: enable_modality_aggregation={self.enable_modality_aggregation}")
            if self.enable_modality_aggregation and num_unique_datasets > 1:
                # Multi-dataset aware aggregation (most sophisticated)
                print("[Server] Performing multi-dataset modality-aware aggregation...")
                global_params, global_summary = self.aggregate_multi_dataset_aware(results)
                if global_params is not None:
                    aggregated_result = (ndarrays_to_parameters(global_params), global_summary)
                else:
                    aggregated_result = None
                    
            elif self.enable_modality_aggregation and num_unique_modalities > 1:
                # Traditional modality-aware aggregation (single dataset, multiple modalities)
                print("[Server] Performing single-dataset modality-aware aggregation...")
                
                # Aggregate within each modality
                modality_results = {}
                for modality in self.modality_groups.keys():
                    modality_params, modality_summary = self.aggregate_within_modality(modality, results)
                    if modality_params is not None:
                        modality_results[modality] = (modality_params, modality_summary)
                
                # Aggregate across modalities
                if modality_results:
                    global_params, global_summary = self.aggregate_across_modalities(modality_results)
                    if global_params is not None:
                        aggregated_result = (ndarrays_to_parameters(global_params), global_summary)
                    else:
                        aggregated_result = None
                else:
                    aggregated_result = None
            else:
                # Traditional FedAvg aggregation
                print(f"[Server] Performing traditional FedAvg aggregation (enable_modality_aggregation={self.enable_modality_aggregation})...")
                aggregated_result = super().aggregate_fit(server_round, results, failures)
                global_summary = {'aggregation_method': 'traditional_fedavg'}

                # Update last_common_param_names from client metadata if available
                if results:
                    param_names_str = results[0][1].metrics.get("param_names_str", "[]")
                    try:
                        self.last_common_param_names = json.loads(param_names_str) if param_names_str else []
                    except json.JSONDecodeError:
                        self.last_common_param_names = []
                
            # Save global model if validation improved
            if client_validation_scores and aggregated_result is not None:
                avg_validation_dice = sum(client_validation_scores) / len(client_validation_scores)
                print(f"[Server] Average validation Dice across clients: {avg_validation_dice:.4f}")
                
                # Save global model if it's the best so far
                if avg_validation_dice > self.best_global_validation_dice:
                    self.best_global_validation_dice = avg_validation_dice
                    self.best_global_round = federated_round
                    print(f"[Server] New best global validation Dice: {avg_validation_dice:.4f}")
                    
                    # Save the global model
                    self._save_best_global_model(aggregated_result[0], federated_round, avg_validation_dice, global_summary)
                else:
                    print(f"[Server] Current validation Dice ({avg_validation_dice:.4f}) <= best ({self.best_global_validation_dice:.4f}), not saving")
                    
            return aggregated_result

    def _save_best_global_model(self, parameters, federated_round: int, validation_dice: float, summary: Dict):
        """Save the best global model in PyTorch format."""
        try:
            import torch
            from datetime import datetime
            
            # Create a metadata-rich checkpoint
            global_checkpoint = {
                'federated_round': federated_round,
                'validation_dice': validation_dice,
                'best_global_validation_dice': self.best_global_validation_dice,
                'timestamp': datetime.now().isoformat(),
                'parameters': parameters,  # Flower's NDArray parameters
                'num_clients': self.expected_num_clients,
                'strategy': 'ModalityAwareFedAvg',
                'model_type': 'nnUNet_global_best',
                'modality_aware_enabled': self.enable_modality_aggregation,
                'modality_groups': dict(self.modality_groups),
                'modality_weights': self.modality_weights,
                'aggregation_summary': summary
            }
            
            # Save path
            save_path = os.path.join(self.global_models_dir, "global_best_model_modality_aware.pt")
            
            # Save the checkpoint
            torch.save(global_checkpoint, save_path)
            
            print(f"[Server] Saved best global model to: {save_path}")
            print(f"[Server] Global model validation Dice: {validation_dice:.4f}")
            print(f"[Server] Global model round: {federated_round}")
            print(f"[Server] Aggregation method: {summary.get('aggregation_method', 'unknown')}")
            
        except Exception as e:
            print(f"[Server] Error saving global model: {e}")


def server_fn(context: Context):
    expected_clients = int(os.environ.get("NUM_CLIENTS", 2))
    training_rounds = int(os.environ.get("NUM_TRAINING_ROUNDS", 3))
    enable_modality_aggregation = os.environ.get("ENABLE_MODALITY_AGGREGATION", "true").lower() == "true"
    
    # Backbone aggregation strategy - always enabled
    enable_backbone_aggregation = True
    
    # Parse modality weights if provided
    modality_weights = None
    if os.environ.get("MODALITY_WEIGHTS"):
        try:
            modality_weights = json.loads(os.environ.get("MODALITY_WEIGHTS"))
        except json.JSONDecodeError:
            print("[Server] Warning: Invalid MODALITY_WEIGHTS JSON, using default weights")
    
    # Total rounds = 2 (preprocessing + initialization) + training_rounds
    total_rounds = 2 + training_rounds
    
    strategy = ModalityAwareFederatedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_available_clients=expected_clients,
        expected_num_clients=expected_clients,
        enable_modality_aggregation=enable_modality_aggregation,
        modality_weights=modality_weights,
    )
    config = ServerConfig(num_rounds=total_rounds)
    
    print(f"[Server] Starting nnUNet federated learning:")
    print(f"[Server] - Expected clients: {expected_clients}")
    print(f"[Server] - Training rounds: {training_rounds}")
    print(f"[Server] - Total rounds: {total_rounds} (2 setup + {training_rounds} training)")
    print(f"[Server] - Modality-aware aggregation: {enable_modality_aggregation}")
    print(f"[Server] - Backbone aggregation strategy: enabled")
    if modality_weights:
        print(f"[Server] - Modality weights: {modality_weights}")
    
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("This is a Modality-Aware Flower ServerApp. Typically run with:")
    print("flower-superlink --server-app=server_app_modality.py:app")