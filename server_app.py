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
            for _, fitres in results:
                client_id = fitres.metrics.get("client_id", "unknown")
                loss = fitres.metrics.get("loss", 0.0)
                epochs = fitres.metrics.get("local_epochs_completed", 0)
                print(f"[Server] Client {client_id}: loss={loss:.4f}, epochs={epochs}")
                
            return super().aggregate_fit(server_round, results, failures)


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
