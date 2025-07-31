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
        
        # Perform weighted averaging within modality
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
        
        # Perform weighted averaging within dataset-modality group
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

    def _weighted_average(self, params_and_weights: List[Tuple[NDArrays, float]]) -> NDArrays:
        """Perform simple weighted average of backbone parameters"""
        if not params_and_weights:
            return []
        
        return self._backbone_weighted_average(params_and_weights)
    
    def _backbone_weighted_average(self, params_and_weights: List[Tuple[NDArrays, float]]) -> NDArrays:
        """Perform simple weighted average of backbone parameters (excludes first/last layers)"""
        print(f"[Server] Performing backbone aggregation for {len(params_and_weights)} clients")
        
        if not params_and_weights:
            return []
        
        # Calculate total weight
        total_weight = sum(weight for _, weight in params_and_weights)
        if total_weight == 0:
            total_weight = len(params_and_weights)  # Equal weighting fallback
        
        # Initialize aggregated parameters
        aggregated_params = []
        num_params = len(params_and_weights[0][0])
        
        print(f"[Server] Aggregating {num_params} backbone parameters from {len(params_and_weights)} clients")
        
        for i in range(num_params):
            # Aggregate i-th parameter across all clients
            weighted_sum = None
            for params, weight in params_and_weights:
                if weighted_sum is None:
                    weighted_sum = (weight / total_weight) * params[i]
                else:
                    weighted_sum += (weight / total_weight) * params[i]
            aggregated_params.append(weighted_sum)
        
        print(f"[Server] Backbone aggregation completed - aggregated {len(aggregated_params)} parameters")
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