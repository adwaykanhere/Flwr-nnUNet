# server_app.py

import os
import flwr as fl
from flwr.common import Context, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from task import merge_dataset_fingerprints


class NnUNetFederatedStrategy(FedAvg):
    """
    FedAvg strategy that implements multi-phase federated learning for nnUNet:
    Round -2: Fingerprint collection (preprocessing)
    Round -1: Global fingerprint distribution (initialization) 
    Round 0+: Regular federated training rounds
    """

    def __init__(self, expected_num_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.expected_num_clients = expected_num_clients
        self.fingerprints_collected: list[dict] = []
        self.global_fingerprint: dict | None = None
        self.current_phase = "preprocessing"  # preprocessing, initialization, training
        
        # Best global model tracking
        self.best_global_validation_dice = 0.0
        self.best_global_round = 0
        self.global_models_dir = "global_models"
        
        # Create global models directory
        import os
        os.makedirs(self.global_models_dir, exist_ok=True)
        print(f"[Server] Global models will be saved to: {self.global_models_dir}")

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
            "local_epochs": 2,  # Default local epochs per round
            "validate": federated_round >= 0,  # Enable validation for training rounds
        }
        
        # Add global fingerprint to config during initialization phase
        # if federated_round == -1 and self.global_fingerprint:
        #     config["global_fingerprint"] = self.global_fingerprint
            
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
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ):
        print(f"[Server] Round {server_round} results: {len(results)} successes, {len(failures)} failures.")
        
        # Determine federated round
        if server_round == 1:
            federated_round = -2  # Preprocessing round
        elif server_round == 2:
            federated_round = -1  # Initialization round
        else:
            federated_round = server_round - 2  # Training rounds

        # Handle preprocessing round - collect fingerprints
        if federated_round == -2:
            print("[Server] Processing fingerprint collection round")
            for _, fitres in results:
                preprocessing_complete = fitres.metrics.get("preprocessing_complete", False)
                fingerprint_cases = fitres.metrics.get("fingerprint_cases", 0)
                if preprocessing_complete:
                    print(f"[Server] Client preprocessed {fingerprint_cases} cases")
                    
            print(f"[Server] Collected {len(results)}/{self.expected_num_clients} fingerprints")
            
            # For now, skip complex fingerprint merging and use simple approach
            if len(results) >= self.expected_num_clients:
                print(f"[Server] All clients completed preprocessing")
            else:
                print(f"[Server] Waiting for more clients ({len(results)}/{self.expected_num_clients})")
                
            # For preprocessing round, return empty aggregation results 
            return None, {}

        # Handle initialization round - distribute global fingerprint and initial model
        elif federated_round == -1:
            print("[Server] Processing initialization round")
            for _, fitres in results:
                init_complete = fitres.metrics.get("initialization_complete", False)
                if init_complete:
                    print(f"[Server] Client initialized successfully")
                    
            # For initialization round, return empty aggregation results 
            return None, {}

        # Handle regular training rounds
        else:
            print(f"[Server] Processing training round {federated_round}")
            
            # Collect client metrics and validation scores
            client_validation_scores = []
            for _, fitres in results:
                client_id = fitres.metrics.get("client_id", "unknown")
                loss = fitres.metrics.get("loss", 0.0)
                epochs = fitres.metrics.get("local_epochs_completed", 0)
                validation_dice = fitres.metrics.get("validation_dice", {})
                
                if isinstance(validation_dice, dict) and "mean" in validation_dice:
                    val_score = validation_dice["mean"]
                    client_validation_scores.append(val_score)
                    print(f"[Server] Client {client_id}: loss={loss:.4f}, epochs={epochs}, val_dice={val_score:.4f}")
                else:
                    print(f"[Server] Client {client_id}: loss={loss:.4f}, epochs={epochs}")
            
            # Perform aggregation
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            # Save global model if validation improved
            if client_validation_scores and aggregated_result is not None:
                # Use average validation score across clients as global metric
                avg_validation_dice = sum(client_validation_scores) / len(client_validation_scores)
                print(f"[Server] Average validation Dice across clients: {avg_validation_dice:.4f}")
                
                # Save global model if it's the best so far
                if avg_validation_dice > self.best_global_validation_dice:
                    self.best_global_validation_dice = avg_validation_dice
                    self.best_global_round = federated_round
                    print(f"[Server] New best global validation Dice: {avg_validation_dice:.4f}")
                    
                    # Save the global model
                    self._save_best_global_model(aggregated_result[0], federated_round, avg_validation_dice)
                else:
                    print(f"[Server] Current validation Dice ({avg_validation_dice:.4f}) <= best ({self.best_global_validation_dice:.4f}), not saving")
                    
            return aggregated_result

    def _save_best_global_model(self, parameters, federated_round: int, validation_dice: float):
        """Save the best global model in PyTorch format."""
        try:
            import torch
            import os
            from datetime import datetime
            
            # Convert Flower parameters to a format suitable for saving
            # Note: This is a simplified approach - in practice, you'd need the actual model architecture
            # to properly reconstruct the state_dict
            
            # Create a metadata-rich checkpoint
            global_checkpoint = {
                'federated_round': federated_round,
                'validation_dice': validation_dice,
                'best_global_validation_dice': self.best_global_validation_dice,
                'timestamp': datetime.now().isoformat(),
                'parameters': parameters,  # Flower's NDArray parameters
                'num_clients': self.expected_num_clients,
                'strategy': 'FedAvg',
                'model_type': 'nnUNet_global_best'
            }
            
            # Save path
            save_path = os.path.join(self.global_models_dir, "global_best_model.pt")
            
            # Save the checkpoint
            torch.save(global_checkpoint, save_path)
            
            print(f"[Server] Saved best global model to: {save_path}")
            print(f"[Server] Global model validation Dice: {validation_dice:.4f}")
            print(f"[Server] Global model round: {federated_round}")
            
        except Exception as e:
            print(f"[Server] Error saving global model: {e}")

    def _parameters_to_state_dict(self, parameters, param_keys=None):
        """Convert Flower parameters back to PyTorch state_dict format."""
        # This would need the parameter keys to reconstruct the state_dict properly
        # For now, we save the raw parameters and metadata
        return {f"param_{i}": param for i, param in enumerate(parameters)}


def server_fn(context: Context):
    expected_clients = int(os.environ.get("NUM_CLIENTS", 1))  # Reduced for testing
    training_rounds = int(os.environ.get("NUM_TRAINING_ROUNDS", 2))  # Reduced for testing
    
    # Total rounds = 2 (preprocessing + initialization) + training_rounds
    total_rounds = 2 + training_rounds
    
    strategy = NnUNetFederatedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_available_clients=expected_clients,
        expected_num_clients=expected_clients,
    )
    config = ServerConfig(num_rounds=total_rounds)
    
    print(f"[Server] Starting nnUNet federated learning:")
    print(f"[Server] - Expected clients: {expected_clients}")
    print(f"[Server] - Training rounds: {training_rounds}")
    print(f"[Server] - Total rounds: {total_rounds} (2 setup + {training_rounds} training)")
    
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("This is a Flower ServerApp. Typically run with:")
    print("flower-supernode --server-app=server_app.py:app")
